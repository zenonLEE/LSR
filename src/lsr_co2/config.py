from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is inconsistent."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str
    data_path: Path
    target_column: str
    feature_columns: tuple[str, ...]
    split_strategy: str
    test_size: float
    split_seed: int
    drop_missing: bool
    split_column: str | None
    train_value: str | None
    test_value: str | None
    k_max: int
    scales: tuple[int, ...]
    rules_path: Path
    prediction_column: str
    xgb_reference: str
    output_min: float
    output_max: float
    short_long_confident: float
    short_long_boundary: float
    short_std_ratio: float
    long_std_threshold: float

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        project_root = config_path.parent.parent

        def project_path(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else project_root / candidate

        config = cls(
            name=str(raw["name"]),
            dataset=str(raw["dataset"]),
            data_path=project_path(raw["data_path"]),
            target_column=str(raw["target_column"]),
            feature_columns=tuple(raw["feature_columns"]),
            split_strategy=str(raw["split_strategy"]),
            test_size=float(raw.get("test_size", 0.2)),
            split_seed=int(raw.get("split_seed", 42)),
            drop_missing=bool(raw.get("drop_missing", False)),
            split_column=raw.get("split_column"),
            train_value=raw.get("train_value"),
            test_value=raw.get("test_value"),
            k_max=int(raw["k_max"]),
            scales=tuple(int(k) for k in raw["scales"]),
            rules_path=project_path(raw["rules_path"]),
            prediction_column=str(raw["prediction_column"]),
            xgb_reference=str(raw["xgb_reference"]),
            output_min=float(raw["output_range"][0]),
            output_max=float(raw["output_range"][1]),
            short_long_confident=float(raw["thresholds"]["short_long_confident"]),
            short_long_boundary=float(raw["thresholds"]["short_long_boundary"]),
            short_std_ratio=float(raw["thresholds"]["short_std_ratio"]),
            long_std_threshold=float(raw["thresholds"]["long_std_threshold"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset not in {"biochar", "dac"}:
            raise ConfigError(f"Unsupported dataset: {self.dataset}")
        if self.split_strategy not in {"random_holdout", "locked_column"}:
            raise ConfigError(f"Unsupported split strategy: {self.split_strategy}")
        if not self.feature_columns:
            raise ConfigError("feature_columns must not be empty")
        if not self.scales or tuple(sorted(set(self.scales))) != self.scales:
            raise ConfigError("scales must be unique and sorted")
        if self.scales[-1] != self.k_max:
            raise ConfigError("the largest scale must equal k_max")
        if self.scales[0] <= 0:
            raise ConfigError("neighbor scales must be positive")
        if self.output_min >= self.output_max:
            raise ConfigError("output_range must be increasing")
        if self.split_strategy == "locked_column" and not all(
            (self.split_column, self.train_value, self.test_value)
        ):
            raise ConfigError("locked_column requires split_column/train_value/test_value")

    def with_split_seed(self, seed: int) -> ExperimentConfig:
        return replace(self, split_seed=seed)

    def with_rules_path(self, path: str | Path) -> ExperimentConfig:
        return replace(self, rules_path=Path(path).resolve())


@dataclass(frozen=True)
class RefinementConfig:
    name: str
    experiment_config: Path
    diagnostic_fraction: float
    diagnostic_seed: int
    neighbor_count: int
    summary_scale: int
    rule_count: int
    minimum_evidence: int
    worst_examples: int
    best_examples: int
    policy_version: str

    @classmethod
    def from_json(cls, path: str | Path) -> RefinementConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        project_root = config_path.parent.parent
        experiment_path = Path(raw["experiment_config"])
        if not experiment_path.is_absolute():
            experiment_path = project_root / experiment_path

        config = cls(
            name=str(raw["name"]),
            experiment_config=experiment_path,
            diagnostic_fraction=float(raw.get("diagnostic_fraction", 0.2)),
            diagnostic_seed=int(raw.get("diagnostic_seed", 43)),
            neighbor_count=int(raw.get("neighbor_count", 15)),
            summary_scale=int(raw.get("summary_scale", 5)),
            rule_count=int(raw.get("rule_count", 3)),
            minimum_evidence=int(raw.get("minimum_evidence", 3)),
            worst_examples=int(raw.get("worst_examples", 10)),
            best_examples=int(raw.get("best_examples", 5)),
            policy_version=str(raw.get("policy_version", "relative-generalizable-v2")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 < self.diagnostic_fraction < 1.0:
            raise ConfigError("diagnostic_fraction must be between zero and one")
        if self.summary_scale <= 0:
            raise ConfigError("summary_scale must be positive")
        if self.neighbor_count < self.summary_scale:
            raise ConfigError("neighbor_count must cover summary_scale")
        if self.rule_count <= 0 or self.minimum_evidence <= 0:
            raise ConfigError("rule_count and minimum_evidence must be positive")
        if self.worst_examples <= 0 or self.best_examples <= 0:
            raise ConfigError("example counts must be positive")
        if not self.policy_version:
            raise ConfigError("policy_version must not be empty")


@dataclass(frozen=True)
class LinearityConfig:
    name: str
    data_path: Path
    feature_columns: tuple[str, ...]
    targets: tuple[tuple[str, str], ...]
    test_size: float
    split_seed: int
    neighbor_count: int
    ridge_alpha: float
    global_xgb_estimators: int
    global_xgb_depth: int
    local_xgb_estimators: int
    local_xgb_depth: int
    model_seed: int
    sensitivity_target: str
    sensitivity_k: tuple[int, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> LinearityConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        project_root = config_path.parent.parent
        data_path = Path(raw["data_path"])
        if not data_path.is_absolute():
            data_path = project_root / data_path
        global_xgb = raw["global_xgb"]
        local_xgb = raw["local_xgb"]
        config = cls(
            name=str(raw["name"]),
            data_path=data_path,
            feature_columns=tuple(str(value) for value in raw["feature_columns"]),
            targets=tuple(
                (str(item["label"]), str(item["column"]))
                for item in raw["targets"]
            ),
            test_size=float(raw.get("test_size", 0.2)),
            split_seed=int(raw.get("split_seed", 42)),
            neighbor_count=int(raw.get("neighbor_count", 30)),
            ridge_alpha=float(raw.get("ridge_alpha", 0.1)),
            global_xgb_estimators=int(global_xgb["n_estimators"]),
            global_xgb_depth=int(global_xgb["max_depth"]),
            local_xgb_estimators=int(local_xgb["n_estimators"]),
            local_xgb_depth=int(local_xgb["max_depth"]),
            model_seed=int(raw.get("model_seed", 42)),
            sensitivity_target=str(raw["sensitivity_target"]),
            sensitivity_k=tuple(int(value) for value in raw["sensitivity_k"]),
        )
        config.validate()
        return config

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(column for _, column in self.targets)

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("linearity name must not be empty")
        if not 0.0 < self.test_size < 1.0:
            raise ConfigError("linearity test_size must be between zero and one")
        if not self.feature_columns or len(set(self.feature_columns)) != len(
            self.feature_columns
        ):
            raise ConfigError("linearity feature_columns must be nonempty and unique")
        if not self.targets or len(set(self.target_columns)) != len(self.targets):
            raise ConfigError("linearity target columns must be nonempty and unique")
        overlap = set(self.feature_columns).intersection(self.target_columns)
        if overlap:
            raise ConfigError(f"linearity features contain target columns: {sorted(overlap)}")
        if self.sensitivity_target not in self.target_columns:
            raise ConfigError("sensitivity_target must be one of the target columns")
        if self.neighbor_count <= 0 or self.ridge_alpha < 0:
            raise ConfigError("neighbor_count must be positive and ridge_alpha nonnegative")
        if not self.sensitivity_k or tuple(sorted(set(self.sensitivity_k))) != self.sensitivity_k:
            raise ConfigError("sensitivity_k must be unique and sorted")
        if self.sensitivity_k[0] <= 0:
            raise ConfigError("sensitivity_k values must be positive")
        if min(
            self.global_xgb_estimators,
            self.global_xgb_depth,
            self.local_xgb_estimators,
            self.local_xgb_depth,
        ) <= 0:
            raise ConfigError("XGBoost sizes and depths must be positive")
