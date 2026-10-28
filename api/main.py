"""D4 FastAPI service: serves the fitted Ridge pipeline selected in D3.

Loads models/pipeline.joblib exactly once, at import time (module-level
globals), so each server process never retrains during serving. If the
artifact fails to load, every endpoint that needs it returns 503 instead
of crashing the process — the server itself still starts and answers.

Run locally:
    uv run uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # `import transformers` must resolve — see models/pipeline.joblib

PIPELINE = None
METADATA = None
LOAD_ERROR = None

try:
    import joblib
    PIPELINE = joblib.load(ROOT / "models" / "pipeline.joblib")
    METADATA = json.loads((ROOT / "models" / "metadata.json").read_text())
except Exception as e:  # artifact missing/corrupt — server still starts, endpoints report 503
    LOAD_ERROR = f"{type(e).__name__}: {e}"

app = FastAPI(
    title="Luxury Resale Ratio API",
    version="1.0",
    description="Predicts the asking-price-to-estimated-retail-price ratio for a SouthPark Mall "
                "brand listing on Rebag. DTSC 3602 D4 — see /docs for schemas.",
)

BRAND_GROUPS = ["Louis Vuitton", "Gucci", "Chanel", "Saint Laurent", "Balenciaga",
                "Tiffany & Co.", "Burberry", "Other SouthPark"]
ITEM_GROUPS = ["Handbag", "Accessory", "Jewelry & Watch", "Shoes"]
AGE_BANDS = ["0-5", "6-10", "11-20", "21+", "Not available"]
CONDITIONS = ["Fair", "Good", "Very Good", "Great", "Excellent", "Pristine"]
COND_RANK = {c: i + 1 for i, c in enumerate(CONDITIONS)}


class PredictRequest(BaseModel):
    brand_group: Literal[tuple(BRAND_GROUPS)] = Field(..., description="SouthPark brand group")
    item_group: Literal[tuple(ITEM_GROUPS)] = Field(..., description="Item category")
    product_type: str = Field(..., min_length=1, max_length=60,
                               description="Rebag sub-category, e.g. 'Totes'. Rare or unknown "
                                           "types are folded into 'Other' automatically.")
    material_group: str = Field("Unknown", max_length=60, description="Exterior material group, or 'Unknown'")
    condition: Literal[tuple(CONDITIONS)] = Field(..., description="Rebag condition grade")
    age_band: Literal[tuple(AGE_BANDS)] = Field("Not available", description="Item age band")
    age_censored: bool = Field(False, description="True for a microchipped item (age may be underestimated)")
    estimated_retail_price: float = Field(..., gt=0, le=200_000, description="Rebag's estimated retail price, USD")
    n_extras: int = Field(0, ge=0, le=10, description="Number of extras listed")
    has_dust_bag: bool = False
    has_strap: bool = False
    has_auth_card: bool = False
    has_box: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "brand_group": "Louis Vuitton", "item_group": "Handbag", "product_type": "Totes",
                "material_group": "Canvas Coated", "condition": "Very Good", "age_band": "6-10",
                "age_censored": False, "estimated_retail_price": 2200, "n_extras": 2,
                "has_dust_bag": True, "has_strap": False, "has_auth_card": True, "has_box": False,
            }
        }
    }


class PredictResponse(BaseModel):
    asking_price_ratio: float = Field(..., description="Predicted asking price as % of estimated retail")
    estimated_asking_price: float = Field(..., description="estimated_retail_price * ratio / 100")
    model_version: str


def _require_pipeline():
    if PIPELINE is None or METADATA is None:
        raise HTTPException(status_code=503, detail=f"Model artifact unavailable: {LOAD_ERROR}")


@app.get("/health")
def health():
    """Liveness check. 503 if the fitted pipeline failed to load."""
    _require_pipeline()
    return {"status": "ok"}


@app.get("/info")
def info():
    """Pipeline metadata: model type, features, training provenance."""
    _require_pipeline()
    return METADATA


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict the asking-price ratio for one listing's known traits.

    Invalid input (bad enum value, non-positive price, etc.) returns 422
    automatically via Pydantic. If the pipeline artifact is unavailable,
    returns 503 rather than a 500.
    """
    _require_pipeline()
    row = pd.DataFrame([{
        "brand_group": req.brand_group,
        "item_group": req.item_group,
        "product_type": req.product_type,
        "material_group": req.material_group,
        "age_band": req.age_band,
        "condition_rank": COND_RANK[req.condition],
        "log_retail_price": float(np.log(req.estimated_retail_price)),
        "n_extras": req.n_extras,
        "age_censored": int(req.age_censored),
        "has_dust_bag": int(req.has_dust_bag),
        "has_strap": int(req.has_strap),
        "has_auth_card": int(req.has_auth_card),
        "has_box": int(req.has_box),
    }])
    pred = float(PIPELINE.predict(row)[0])
    pred = max(0.0, pred)  # the pipeline has never produced a negative value, but guard regardless
    return PredictResponse(
        asking_price_ratio=round(pred, 2),
        estimated_asking_price=round(req.estimated_retail_price * pred / 100, 2),
        model_version=METADATA.get("git_commit") or "unknown",
    )
