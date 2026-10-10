"""Inference wrapper used by the selection engine."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cellselect.data.features import FEATURE_COLUMNS

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when no usable model artifact exists."""


@dataclass
class Prediction:
    probability: np.ndarray
    uncertainty: np.ndarray | None  # std of per-tree probabilities (ensembles only)


class FailurePredictor:
    def __init__(self, pipeline, name: str, threshold: float, card: dict | None = None) -> None:
        self.pipeline = pipeline
        self.name = name
        self.threshold = threshold
        self.card = card or {}

    @classmethod
    def load(cls, models_dir: str | Path, name: str | None = None) -> FailurePredictor:
        models_dir = Path(models_dir)
        card_path = models_dir / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        name = name or card.get("best_model")
        if not name:
            raise ModelUnavailableError(f"no model card / best model found in {models_dir}")
        path = models_dir / f"{name}.joblib"
        if not path.exists():
            raise ModelUnavailableError(f"model artifact {path} not found - run `cellselect train` first")
        try:
            blob = joblib.load(path)
        except Exception as exc:  # corrupted / incompatible pickle
            raise ModelUnavailableError(f"cannot load {path}: {exc}") from exc
        if list(blob.get("features", [])) != FEATURE_COLUMNS:
            raise ModelUnavailableError(f"model {path} was trained with a different feature set")
        return cls(blob["pipeline"], name, float(blob.get("threshold", 0.5)), card)

    @property
    def seen_change_types(self) -> set[str]:
        return set(self.card.get("seen_change_types", []))

    @property
    def seen_change_areas(self) -> set[str]:
        return set(self.card.get("seen_change_areas", []))

    def predict(self, features: pd.DataFrame) -> Prediction:
        X = features[FEATURE_COLUMNS]
        proba = self.pipeline.predict_proba(X)[:, 1]
        clf = self.pipeline.named_steps["clf"]
        std = None
        if hasattr(clf, "estimators_") and isinstance(getattr(clf, "estimators_", None), list):
            Xt = self.pipeline.named_steps["prep"].transform(X)
            per_tree = np.stack(
                [
                    est.predict_proba(Xt)[:, 1] if est.n_classes_ == 2 else np.zeros(len(X))
                    for est in clf.estimators_
                ]
            )
            std = per_tree.std(axis=0)
        return Prediction(proba, std)
