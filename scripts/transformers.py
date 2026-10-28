"""Custom transformers for the D3/D4 modeling pipeline.

Kept in its own module (not inline in a notebook or script) because joblib
pickles a *reference* to the class's import path, not its source — the
training script and the FastAPI serving code must both be able to
`from scripts.transformers import RareCategoryGrouper` and get the same
class, or `joblib.load` fails at serve time.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Fold categories seen fewer than `min_count` times in training into "Other".

    Why this is needed: unlike `brand_group` and `material_group` (already
    grouped during D2 cleaning), `product_type` has 25 raw categories and
    several appear only once or twice in the training data (e.g. a single
    "Hats & Hair Accessories" row). Left alone, those create one-hot columns
    that are pure noise for a linear model and can destabilize a tree split.
    Worse, a product type never seen in training at all is a hard failure
    for `OneHotEncoder` without `handle_unknown="ignore"` — and even with
    that, an unseen category just becomes an all-zero row with no signal.

    This transformer is fit only on the training fold (the "keep" set is
    learned, never looked up at transform time), so it never leaks
    validation/test information and it gracefully folds both rare *and*
    entirely unseen categories into "Other" at predict time.
    """

    def __init__(self, min_count=10):
        self.min_count = min_count

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.keep_ = {
            col: set(X[col].value_counts()[lambda s: s >= self.min_count].index)
            for col in self.columns_
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        for col in self.columns_:
            X[col] = X[col].where(X[col].isin(self.keep_[col]), "Other")
        return X

    def get_feature_names_out(self, input_features=None):
        return pd.Index(self.columns_)
