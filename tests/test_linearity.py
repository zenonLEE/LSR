from pathlib import Path

import numpy as np
import pytest

from lsr_co2.config import LinearityConfig
from lsr_co2.linearity import (
    LinearityRun,
    load_linearity_frame,
    run_linearity_analysis,
    split_linearity_indices,
)


@pytest.fixture
def reproduced_run(
    linearity_config: LinearityConfig,
    tmp_path: Path,
) -> LinearityRun:
    pytest.importorskip("xgboost")
    return run_linearity_analysis(
        linearity_config,
        tmp_path / "linearity",
        save_figure=False,
    )


def test_linearity_protocol_excludes_all_targets(
    linearity_config: LinearityConfig,
) -> None:
    assert len(linearity_config.feature_columns) == 19
    assert set(linearity_config.feature_columns).isdisjoint(
        linearity_config.target_columns
    )
    frame = load_linearity_frame(linearity_config)
    train, test = split_linearity_indices(len(frame), linearity_config)
    assert frame.shape == (120, 23)
    assert (len(train), len(test)) == (96, 24)


def test_linearity_run_writes_local_artifacts(reproduced_run: LinearityRun) -> None:
    output = Path(reproduced_run.summary.output_dir)
    expected = {
        "local_linearity_stats.csv",
        "local_vs_global_comparison.csv",
        "k_sensitivity_analysis.csv",
        "linearity.metadata.json",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    assert reproduced_run.summary.train_size == 96
    assert reproduced_run.summary.test_size == 24


def test_linearity_metrics_use_full_precision_formula(
    reproduced_run: LinearityRun,
) -> None:
    comparison = reproduced_run.model_comparison
    expected_gain = (
        100.0
        * (comparison["Local_Linear"] - comparison["Global_Linear"])
        / comparison["Global_Linear"]
    )
    np.testing.assert_allclose(comparison["Relative_Gain_Pct"], expected_gain)
    assert reproduced_run.summary.local_linear_to_nonlinear_ratio == pytest.approx(
        comparison["Local_Linear"].mean()
        / comparison["Local_Nonlinear"].mean()
    )
    assert np.isfinite(comparison.select_dtypes(include=["number"])).all().all()


def test_support_scores_and_sensitivity_are_complete(
    linearity_config: LinearityConfig,
    reproduced_run: LinearityRun,
) -> None:
    assert reproduced_run.support_statistics["Target"].tolist() == [
        label for label, _ in linearity_config.targets
    ]
    assert all(len(scores) == 24 for scores in reproduced_run.local_scores.values())
    assert reproduced_run.k_sensitivity["k"].tolist() == list(
        linearity_config.sensitivity_k
    )
