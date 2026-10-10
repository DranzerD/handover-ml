from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cellselect.data.generate import GenerationConfig, generate_history
from cellselect.ml.train import TrainConfig, train_models


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small but complete synthetic history (shared by several tests)."""
    out = tmp_path_factory.mktemp("history")
    generate_history(GenerationConfig(seed=3, n_changes=45, new_tests={"HO-PP": 0.8}), out)
    return out


@pytest.fixture(scope="session")
def small_models(small_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("models")
    executions = pd.read_csv(small_dataset / "executions.csv")
    train_models(
        executions, TrainConfig(cv_folds=2, fast=True, models=["logistic_regression", "random_forest"]), out
    )
    return out
