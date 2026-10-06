from pathlib import Path

import pytest

from lsr_co2.security import (
    contains_potential_secret,
    redact_sensitive_text,
    validate_private_output_path,
)


def test_secret_shapes_are_detected_and_redacted() -> None:
    fake_key = "sk-" + "not-a-real-key-1234567890"
    payload = f"API_KEY={fake_key}"
    assert contains_potential_secret(payload)
    redacted = redact_sensitive_text(payload)
    assert fake_key not in redacted
    assert "[REDACTED]" in redacted


def test_repository_output_must_be_below_artifacts(tmp_path: Path) -> None:
    allowed = validate_private_output_path(
        tmp_path / "artifacts" / "run.csv", project_root=tmp_path
    )
    assert allowed == (tmp_path / "artifacts" / "run.csv").resolve()
    with pytest.raises(ValueError, match="under artifacts"):
        validate_private_output_path(tmp_path / "results.csv", project_root=tmp_path)


def test_output_outside_repository_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    output = validate_private_output_path(tmp_path / "private-run.csv", project_root=root)
    assert output == (tmp_path / "private-run.csv").resolve()
