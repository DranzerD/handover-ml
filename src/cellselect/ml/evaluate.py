"""Model evaluation: held-out metrics tables, PR curves, confusion matrices, permutation importance."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from cellselect.data.features import FEATURE_COLUMNS, TARGET
from cellselect.ml.predictor import FailurePredictor
from cellselect.ml.train import temporal_split
from cellselect.reporting import charts

log = logging.getLogger(__name__)


def metrics_table(card: dict) -> pd.DataFrame:
    rows = []
    for name, m in card["test"].items():
        rows.append(
            {
                "model": name,
                "cv_pr_auc": card["cv"].get(name, {}).get("pr_auc"),
                "test_pr_auc": m["pr_auc"],
                "test_roc_auc": m["roc_auc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "threshold": m["threshold"],
                **{k: v for k, v in m["confusion_matrix"].items()},
            }
        )
    return pd.DataFrame(rows)


def evaluate_models(
    data_dir: str | Path,
    models_dir: str | Path,
    out_dir: str | Path,
    importance_rows: int = 6000,
    seed: int = 0,
) -> dict:
    models_dir, out = Path(models_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    card = json.loads((models_dir / "model_card.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(models_dir / "test_predictions.csv")
    table = metrics_table(card)
    table.to_csv(out / "model_metrics.csv", index=False)
    plots = {"pr_curves": charts.plot_pr_curves(preds, out / "pr_curves.png", list(card["test"]))}
    best = card["best_model"]
    plots["confusion_best"] = charts.plot_confusion(
        card["test"][best]["confusion_matrix"],
        f"{best.replace('_', ' ')} @ threshold {card['test'][best]['threshold']:.2f}",
        out / "confusion_best.png",
    )
    executions = pd.read_csv(Path(data_dir) / "executions.csv")
    _, test_df, _ = temporal_split(executions, card["config"]["train_fraction"])
    sample = test_df.sample(min(importance_rows, len(test_df)), random_state=seed)
    predictor = FailurePredictor.load(models_dir, best)
    log.info("computing permutation importance for %s on %d held-out rows", best, len(sample))
    pi = permutation_importance(
        predictor.pipeline,
        sample[FEATURE_COLUMNS],
        sample[TARGET],
        scoring="average_precision",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    imp = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    plots["feature_importance"] = charts.plot_feature_importance(imp, out / "feature_importance.png")
    summary = {
        "best_model": best,
        "metrics": table.to_dict(orient="records"),
        "top_features": imp.head(10).to_dict(orient="records"),
        "plots": {k: str(v) for k, v in plots.items()},
        "synthetic_data": True,
    }
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
