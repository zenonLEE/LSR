from __future__ import annotations

import re
from pathlib import Path

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:gh[opurs]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*", re.IGNORECASE),
    re.compile(
        r"(?im)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\s*[:=]\s*['\"]?[^\s'\",;]{8,}"
    ),
)


def contains_potential_secret(value: str) -> bool:
    """Return true for high-confidence credential shapes without exposing the value."""
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def redact_sensitive_text(value: str) -> str:
    """Redact high-confidence credential shapes from logs and saved raw responses."""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def validate_private_output_path(
    value: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Require repository-local generated output to live below ignored artifacts/."""
    root = Path(project_root or Path.cwd()).resolve()
    output = Path(value).resolve()
    artifacts = (root / "artifacts").resolve()
    if output.is_relative_to(root) and not output.is_relative_to(artifacts):
        raise ValueError(
            "Repository-local generated output must be placed under artifacts/ "
            "to reduce accidental commits"
        )
    return output
