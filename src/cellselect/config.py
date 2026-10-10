"""Experiment configuration (JSON) shared by the CLI and the benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from cellselect.data.generate import GenerationConfig
from cellselect.ml.train import TrainConfig
from cellselect.selection.engine import ABLATION_NAMES, STRATEGY_NAMES, SelectionPolicy


@dataclass
class BenchmarkSettings:
    budgets: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    strategies: list[str] = field(default_factory=lambda: list(STRATEGY_NAMES))
    random_repeats: int = 5
    max_test_changes: int | None = None


@dataclass
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    selection: SelectionPolicy = field(default_factory=SelectionPolicy)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, data: dict | None):
    data = data or {}
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    cfg = ExperimentConfig(
        seeds=list(raw.get("seeds", [0, 1, 2])),
        generation=_build(GenerationConfig, raw.get("generation")),
        training=_build(TrainConfig, raw.get("training")),
        selection=_build(SelectionPolicy, raw.get("selection")),
        benchmark=_build(BenchmarkSettings, raw.get("benchmark")),
        jobs=int(raw.get("jobs", 1)),
    )
    for s in cfg.benchmark.strategies:
        if s not in STRATEGY_NAMES + ABLATION_NAMES:
            raise ValueError(f"unknown strategy {s!r} in config")
    for b in cfg.benchmark.budgets:
        if not 0 < b <= 1:
            raise ValueError(f"budget fraction {b} must be in (0, 1]")
    return cfg
