from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lsr_co2.config import ExperimentConfig, RefinementConfig
from lsr_co2.datasets import load_dataset, split_dataset
from lsr_co2.providers import CompletionResponse, PredictionResponse, TokenUsage
from lsr_co2.refinement import (
    RefinementError,
    load_diagnostic_predictions,
    run_residual_guided_refinement,
    screen_rule_proposal,
    split_outer_training,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeRefinementProvider:
    def __init__(self) -> None:
        self.prediction_prompts: list[str] = []
        self.completion_prompts: list[str] = []

    def predict(self, prompt: str) -> PredictionResponse:
        self.prediction_prompts.append(prompt)
        return PredictionResponse(
            value=0.5,
            text="0.5",
            usage=TokenUsage(input_tokens=10, output_tokens=1),
            raw_output="numeric-response",
        )

    def complete(self, prompt: str) -> CompletionResponse:
        self.completion_prompts.append(prompt)
        rules = """RULE 1: When a broad dry condition holds, adjust toward the local mean.
Evidence: ids [1, 2, 3].
Physics: Accessible amine sites control dry uptake.
Constraints: A=yes B=yes C=yes D=yes.

RULE 2: When pore volume is low, shrink relative to the local mean.
Evidence: ids [4, 5, 6].
Physics: Pore blockage reduces accessible sites.
Constraints: A=yes B=yes C=yes D=yes.

RULE 3: When local statistics agree, use only a small relative correction.
Evidence: ids [7, 8, 9].
Physics: Consistent neighborhoods support conservative adjustment.
Constraints: A=yes B=yes C=yes D=yes.
"""
        return CompletionResponse(
            text=rules,
            usage=TokenUsage(input_tokens=20, output_tokens=10),
            raw_output="rule-response",
        )


def _write_diagnostic_input(
    path: Path,
    diagnostic: pd.DataFrame,
    experiment: ExperimentConfig,
) -> Path:
    frame = diagnostic.copy()
    frame["v1_pred"] = (
        frame[experiment.target_column].to_numpy(dtype=float)
        + np.linspace(-0.15, 0.15, len(frame))
    )
    frame.to_csv(path, index=False)
    return path


def test_dac_inner_split_and_local_diagnostics_match(
    dac_configs: tuple[ExperimentConfig, RefinementConfig],
    tmp_path: Path,
) -> None:
    experiment, refinement = dac_configs
    outer = split_dataset(load_dataset(experiment), experiment)
    inner = split_outer_training(outer.train, refinement)
    assert (len(outer.train), len(inner.sub_train), len(inner.diagnostic)) == (117, 93, 24)
    diagnostic_path = _write_diagnostic_input(
        tmp_path / "diagnostics.csv", inner.diagnostic, experiment
    )
    diagnostics = load_diagnostic_predictions(
        diagnostic_path,
        inner.diagnostic,
        experiment,
    )
    assert len(diagnostics) == 24
    assert diagnostics["diagnostic_id"].tolist() == list(range(1, 25))


def test_offline_end_to_end_refinement(
    dac_configs: tuple[ExperimentConfig, RefinementConfig],
    tmp_path: Path,
) -> None:
    _, refinement = dac_configs
    provider = FakeRefinementProvider()
    summary = run_residual_guided_refinement(
        refinement,
        provider,
        tmp_path,
        model_name="fake-model",
    )
    assert len(provider.prediction_prompts) == 24
    assert len(provider.completion_prompts) == 1
    assert all("## Diagnostic sample" in prompt for prompt in provider.prediction_prompts)
    assert all("xgb_reference_prediction" not in prompt for prompt in provider.prediction_prompts)
    assert summary.sub_train_size == 93
    assert summary.diagnostic_size == 24
    assert Path(summary.frozen_rules_path).exists()
    assert '"passed": true' in (tmp_path / "screening.json").read_text(
        encoding="utf-8"
    )


def test_reuse_local_diagnostics_skips_numeric_calls(
    dac_configs: tuple[ExperimentConfig, RefinementConfig],
    tmp_path: Path,
) -> None:
    experiment, refinement = dac_configs
    outer = split_dataset(load_dataset(experiment), experiment)
    inner = split_outer_training(outer.train, refinement)
    diagnostic_path = _write_diagnostic_input(
        tmp_path / "diagnostics.csv", inner.diagnostic, experiment
    )
    provider = FakeRefinementProvider()
    summary = run_residual_guided_refinement(
        refinement,
        provider,
        tmp_path,
        model_name="fake-model",
        diagnostic_input=diagnostic_path,
        save_prompt=True,
    )
    assert provider.prediction_prompts == []
    assert summary.diagnostic_source.endswith("diagnostics.csv")
    assert (tmp_path / "rule_proposal_prompt.txt").exists()


def test_rule_screening_rejects_bad_evidence() -> None:
    rules = """RULE 1: condition
Evidence: ids [1, 99].
Physics: reason.
Constraints: A=yes B=yes C=yes D=yes.
"""
    result = screen_rule_proposal(
        rules,
        rule_count=1,
        minimum_evidence=3,
        diagnostic_size=24,
    )
    assert not result.passed
    assert any("minimum" in failure for failure in result.failures)
    assert any("unknown" in failure for failure in result.failures)


def test_rule_screening_rejects_potential_credentials() -> None:
    fake_key = "sk-" + "not-a-real-key-1234567890"
    rules = f"""RULE 1: When broad conditions hold, adjust toward the local mean.
Evidence: ids [1, 2, 3].
Physics: Accessible sites control uptake.
Constraints: A=yes B=yes C=yes D=yes.
API_KEY={fake_key}
"""
    result = screen_rule_proposal(
        rules,
        rule_count=1,
        minimum_evidence=3,
        diagnostic_size=24,
    )
    assert not result.passed
    assert any("potential credential" in failure for failure in result.failures)


def test_failed_rule_screening_does_not_persist_candidate(
    dac_configs: tuple[ExperimentConfig, RefinementConfig],
    tmp_path: Path,
) -> None:
    class InvalidRuleProvider(FakeRefinementProvider):
        def complete(self, prompt: str) -> CompletionResponse:
            return CompletionResponse(
                text="not a valid rule proposal",
                usage=TokenUsage(),
                raw_output="invalid-response",
            )

    _, refinement = dac_configs
    with pytest.raises(RefinementError, match="failed structural screening"):
        run_residual_guided_refinement(
            refinement,
            InvalidRuleProvider(),
            tmp_path,
            model_name="fake-model",
        )
    assert not (tmp_path / "candidate_rules.md").exists()
    assert not (tmp_path / "frozen_rules.md").exists()


def test_versioned_dac_rules_pass_structural_screening() -> None:
    result = screen_rule_proposal(
        (ROOT / "rules/dac_clean_v2.md").read_text(encoding="utf-8"),
        rule_count=3,
        minimum_evidence=3,
        diagnostic_size=24,
    )
    assert result.passed
    assert result.evidence_ids == {
        "1": [1, 11, 12, 18, 19],
        "2": [7, 23, 24],
        "3": [9, 13, 15, 21],
    }


def test_diagnostic_input_row_mismatch_is_rejected(
    dac_configs: tuple[ExperimentConfig, RefinementConfig],
    tmp_path: Path,
) -> None:
    experiment, refinement = dac_configs
    outer = split_dataset(load_dataset(experiment), experiment)
    inner = split_outer_training(outer.train, refinement)
    source_path = _write_diagnostic_input(
        tmp_path / "diagnostics.csv", inner.diagnostic, experiment
    )
    source = pd.read_csv(source_path)
    bad_path = tmp_path / "bad.csv"
    source.iloc[:-1].to_csv(bad_path, index=False)
    try:
        load_diagnostic_predictions(bad_path, inner.diagnostic, experiment)
    except Exception as error:
        assert "23 rows; expected 24" in str(error)
    else:
        raise AssertionError("mismatched diagnostic input was accepted")
