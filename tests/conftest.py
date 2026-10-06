from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lsr_co2.config import ExperimentConfig, LinearityConfig, RefinementConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_biochar_csv(tmp_path: Path) -> Path:
    config = LinearityConfig.from_json(ROOT / "configs/biochar_linearity.json")
    rng = np.random.default_rng(20260822)
    features = rng.normal(size=(120, len(config.feature_columns)))
    frame = pd.DataFrame(features, columns=config.feature_columns)
    frame["SBET"] = 450.0 + 60.0 * features[:, 0] - 25.0 * features[:, 3]
    frame["Vtotal"] = 0.7 + 0.12 * features[:, 1] + 0.04 * features[:, 5]
    frame["Vnarrow"] = 0.3 + 0.08 * features[:, 2] - 0.03 * features[:, 6]
    frame["CO2_uptake"] = (
        2.0 + 0.25 * features[:, 4] + 0.12 * features[:, 8] + rng.normal(0, 0.05, 120)
    )
    path = tmp_path / "biochar.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def biochar_config(synthetic_biochar_csv: Path) -> ExperimentConfig:
    base = ExperimentConfig.from_json(ROOT / "configs/biochar.json")
    return replace(base, data_path=synthetic_biochar_csv)


@pytest.fixture
def linearity_config(synthetic_biochar_csv: Path) -> LinearityConfig:
    base = LinearityConfig.from_json(ROOT / "configs/biochar_linearity.json")
    return replace(base, data_path=synthetic_biochar_csv)


@pytest.fixture
def synthetic_dac_csv(tmp_path: Path) -> Path:
    base = ExperimentConfig.from_json(ROOT / "configs/dac.json")
    rng = np.random.default_rng(20260823)
    rows = 147
    frame = pd.DataFrame(
        rng.normal(size=(rows, len(base.feature_columns))),
        columns=base.feature_columns,
    )
    frame["loading"] = rng.uniform(0.1, 2.0, rows)
    frame["primary_amine"] = rng.uniform(0.0, 1.0, rows)
    frame["secondary_amine"] = rng.uniform(0.0, 1.0, rows)
    frame["tertiary_amine"] = rng.uniform(0.0, 1.0, rows)
    frame["N_content"] = rng.uniform(1.0, 15.0, rows)
    frame["Vpore_after"] = rng.uniform(0.05, 1.5, rows)
    frame["temp_C"] = rng.integers(20, 81, rows)
    frame["CO2_ppm"] = rng.integers(300, 1001, rows)
    frame["RH"] = rng.integers(0, 2, rows)
    frame["amine_code"] = rng.integers(0, 4, rows)
    target = (
        0.3
        + 0.8 * frame["loading"].to_numpy()
        + 0.03 * frame["N_content"].to_numpy()
        + rng.normal(0, 0.04, rows)
    )
    frame["CO2_uptake"] = target
    frame["split"] = ["train"] * 117 + ["test"] * 30
    frame["xgb_pred"] = target + rng.normal(0, 0.08, rows)
    frame["amine_clean"] = [f"amine-{index % 4}" for index in range(rows)]
    path = tmp_path / "dac.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def dac_configs(
    tmp_path: Path,
    synthetic_dac_csv: Path,
) -> tuple[ExperimentConfig, RefinementConfig]:
    with (ROOT / "configs/dac.json").open(encoding="utf-8") as handle:
        experiment_payload = json.load(handle)
    experiment_payload["data_path"] = str(synthetic_dac_csv)
    experiment_payload["rules_path"] = str(ROOT / "rules/dac_clean_v2.md")
    experiment_path = tmp_path / "dac.json"
    experiment_path.write_text(
        json.dumps(experiment_payload, indent=2), encoding="utf-8"
    )

    experiment = ExperimentConfig.from_json(experiment_path)
    refinement = replace(
        RefinementConfig.from_json(ROOT / "configs/dac_refinement.json"),
        experiment_config=experiment_path,
    )
    return experiment, refinement
