"""Model zoo and preprocessing pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cellselect.data.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting")


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", numeric, NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_model(name: str, seed: int = 0, fast: bool = False) -> Pipeline:
    """Build an unfitted preprocessing + classifier pipeline."""
    if name == "logistic_regression":
        clf = LogisticRegression(C=0.5, max_iter=3000)
        return Pipeline([("prep", _preprocessor(scale=True)), ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=120 if fast else 300,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("prep", _preprocessor(scale=False)), ("clf", clf)])
    if name == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=150 if fast else 300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            min_samples_leaf=20,
            random_state=seed,
        )
        return Pipeline([("prep", _preprocessor(scale=False)), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
