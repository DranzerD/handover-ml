"""Model training with a temporal hold-out and forward-chaining cross-validation.

Leakage controls:
* rows are grouped by change; train/validation/test splits never share a change,
* splits are chronological (train on the past, evaluate on the future),
* features were computed from strictly earlier changes (see ``data.features``),
* the decision threshold is tuned on validation folds, never on the test period.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cellselect.data.features import FEATURE_COLUMNS, TARGET
from cellselect.ml.models import MODEL_NAMES, make_model

log = logging.getLogger(__name__)

# Simple non-ML scoring heuristics reported alongside the models for context.
HEURISTIC_SCORES = {
    "heuristic_hist_fail_rate": lambda df: df["hist_fail_rate"].to_numpy(),
    "heuristic_last_failed": lambda df: df["last_failed"].to_numpy() + 0.01 * df["hist_fail_rate"].to_numpy(),
    "heuristic_change_overlap": lambda df: (
        df["overlap_frac"].to_numpy() + 0.01 * df["hist_fail_rate"].to_numpy()
    ),
}


@dataclass
class TrainConfig:
    train_fraction: float = 0.7
    cv_folds: int = 4
    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    seed: int = 0
    selection_metric: str = "pr_auc"
    fast: bool = False


def temporal_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    changes = np.sort(df["change_id"].unique())
    if len(changes) < 5:
        raise ValueError("need at least 5 changes to split")
    cutoff = int(changes[int(len(changes) * train_fraction)])
    return df[df["change_id"] < cutoff], df[df["change_id"] >= cutoff], cutoff


def forward_chaining_folds(
    change_ids: np.ndarray, k: int, min_train_frac: float = 0.4
) -> list[tuple[np.ndarray, np.ndarray]]:
    ids = np.sort(np.unique(change_ids))
    start = int(len(ids) * min_train_frac)
    bounds = np.linspace(start, len(ids), k + 1).astype(int)
    return [(ids[: bounds[i]], ids[bounds[i] : bounds[i + 1]]) for i in range(k) if bounds[i + 1] > bounds[i]]


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _fit(name: str, X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig):
    model = make_model(name, cfg.seed, fast=cfg.fast)
    model.fit(X, y)
    return model


def train_models(executions: pd.DataFrame, cfg: TrainConfig, out_dir: str | Path) -> dict:
    """Train all configured models, pick the best by CV metric, evaluate on the temporal hold-out."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df, test_df, cutoff = temporal_split(executions, cfg.train_fraction)
    log.info(
        "train: %d rows (%d changes), test: %d rows (%d changes), cutoff change %d",
        len(train_df),
        train_df.change_id.nunique(),
        len(test_df),
        test_df.change_id.nunique(),
        cutoff,
    )
    X_tr, y_tr = train_df[FEATURE_COLUMNS], train_df[TARGET].to_numpy()
    X_te, y_te = test_df[FEATURE_COLUMNS], test_df[TARGET].to_numpy()
    folds = forward_chaining_folds(train_df["change_id"].to_numpy(), cfg.cv_folds)

    card: dict = {
        "synthetic_data": True,
        "train_changes": [int(train_df.change_id.min()), int(train_df.change_id.max())],
        "test_changes": [int(test_df.change_id.min()), int(test_df.change_id.max())],
        "cutoff_change_id": cutoff,
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rate": float(y_tr.mean()),
        "test_positive_rate": float(y_te.mean()),
        "cv": {},
        "test": {},
        "sklearn_version": sklearn.__version__,
        "config": asdict(cfg),
        "seen_change_types": sorted(train_df["change_type"].unique().tolist()),
        "seen_change_areas": sorted(train_df["change_area"].unique().tolist()),
    }
    preds = test_df[["change_id", "test_id", TARGET]].copy()
    for name in cfg.models:
        t0 = time.perf_counter()
        oof_y, oof_p = [], []
        for tr_ids, va_ids in folds:
            tr = train_df[train_df.change_id.isin(tr_ids)]
            va = train_df[train_df.change_id.isin(va_ids)]
            m = _fit(name, tr[FEATURE_COLUMNS], tr[TARGET].to_numpy(), cfg)
            oof_p.append(m.predict_proba(va[FEATURE_COLUMNS])[:, 1])
            oof_y.append(va[TARGET].to_numpy())
        oy, op = np.concatenate(oof_y), np.concatenate(oof_p)
        threshold = best_f1_threshold(oy, op)
        card["cv"][name] = classification_metrics(oy, op, threshold)
        model = _fit(name, X_tr, y_tr, cfg)
        p_te = model.predict_proba(X_te)[:, 1]
        preds[f"p_{name}"] = p_te
        card["test"][name] = classification_metrics(y_te, p_te, threshold)
        card["test"][name]["fit_seconds"] = round(time.perf_counter() - t0, 2)
        joblib.dump(
            {"pipeline": model, "name": name, "threshold": threshold, "features": FEATURE_COLUMNS},
            out / f"{name}.joblib",
        )
        log.info(
            "%-20s CV PR-AUC %.3f | test PR-AUC %.3f ROC-AUC %.3f F1 %.3f (%.1fs)",
            name,
            card["cv"][name]["pr_auc"] or float("nan"),
            card["test"][name]["pr_auc"] or float("nan"),
            card["test"][name]["roc_auc"] or float("nan"),
            card["test"][name]["f1"],
            card["test"][name]["fit_seconds"],
        )
    for hname, fn in HEURISTIC_SCORES.items():
        tr_score = fn(train_df)
        thr = best_f1_threshold(y_tr, tr_score)
        card["test"][hname] = classification_metrics(y_te, fn(test_df), thr)
        preds[f"p_{hname}"] = fn(test_df)
    best = max(cfg.models, key=lambda n: card["cv"][n][cfg.selection_metric] or -1.0)
    card["best_model"] = best
    card["best_threshold"] = card["cv"][best]["threshold"]
    preds.to_csv(out / "test_predictions.csv", index=False)
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    log.info("best model by CV %s: %s", cfg.selection_metric, best)
    return card
