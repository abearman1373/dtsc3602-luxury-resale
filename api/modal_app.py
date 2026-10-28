"""Deploy the D4 FastAPI service on Modal.

Pins scikit-learn (and pandas/numpy) to the exact versions used to fit
models/pipeline.joblib, per the course's "exact scikit-learn version
matching" requirement — a version mismatch can silently change or break
unpickling of a fitted estimator.

One-time setup (PowerShell):
    uv run modal setup
Deploy:
    uv run modal deploy api/modal_app.py
Modal prints the public URL and /docs is at <url>/docs.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi==0.162.0", "uvicorn==0.37.0", "pydantic==2.14.1",
        "scikit-learn==1.9.1", "pandas==3.1.0", "numpy==2.5.4", "joblib>=1.4",
    )
    .add_local_file(ROOT / "api" / "main.py", "/root/api/main.py")
    .add_local_file(ROOT / "api" / "__init__.py", "/root/api/__init__.py")
    .add_local_file(ROOT / "scripts" / "transformers.py", "/root/scripts/transformers.py")
    .add_local_file(ROOT / "models" / "pipeline.joblib", "/root/models/pipeline.joblib")
    .add_local_file(ROOT / "models" / "metadata.json", "/root/models/metadata.json")
)
app = modal.App("luxury-resale-api", image=image)


@app.function(max_containers=1)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from api.main import app as fastapi_instance
    return fastapi_instance
