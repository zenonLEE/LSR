# ruff: noqa: B904
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .security import redact_sensitive_text

_SAFE_ENVIRONMENT_NAMES = {
    "APPDATA",
    "CODEX_HOME",
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
}


class ProviderError(RuntimeError):
    """Raised when a prediction provider does not return a usable response."""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class PredictionResponse:
    value: float
    text: str
    usage: TokenUsage
    raw_output: str


@dataclass(frozen=True)
class CompletionResponse:
    text: str
    usage: TokenUsage
    raw_output: str


class PredictionProvider(Protocol):
    def predict(self, prompt: str) -> PredictionResponse: ...


class TextProvider(Protocol):
    def complete(self, prompt: str) -> CompletionResponse: ...


class CodexCliProvider:
    """Run numeric predictions or text completions through a local Codex CLI."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str = "xhigh",
        executable: str | Path | None = None,
        timeout_seconds: int = 600,
        output_min: float = 0.0,
        output_max: float = 10.0,
    ) -> None:
        self.executable = str(executable) if executable else _find_codex()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.output_min = output_min
        self.output_max = output_max

    def predict(self, prompt: str) -> PredictionResponse:
        completion = self.complete(prompt)
        value = parse_numeric_prediction(completion.text)
        if not self.output_min <= value <= self.output_max:
            raise ProviderError(
                f"Prediction {value} is outside [{self.output_min}, {self.output_max}]"
            )
        return PredictionResponse(
            value=value,
            text=completion.text,
            usage=completion.usage,
            raw_output=completion.raw_output,
        )

    def complete(self, prompt: str) -> CompletionResponse:
        with tempfile.TemporaryDirectory(prefix="lsr-co2-codex-") as workspace:
            command = [
                self.executable,
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--strict-config",
                "--skip-git-repo-check",
                "--ephemeral",
                "--cd",
                workspace,
                "-c",
                "model_provider=openai",
                "-c",
                f"model_reasoning_effort={self.reasoning_effort}",
                "-c",
                "shell_environment_policy.inherit=none",
                "-c",
                "shell_environment_policy.ignore_default_excludes=false",
                "-c",
                "default_permissions=lsr_co2_private",
                "-c",
                "permissions.lsr_co2_private="
                '{filesystem={":minimal"="read",'
                '":workspace_roots"={"."="read"}},network={enabled=false}}',
                "-m",
                self.model,
                "--json",
                "-",
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=workspace,
                    env=_safe_codex_environment(),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise ProviderError(
                    f"Codex CLI timed out after {self.timeout_seconds}s"
                ) from error

        if completed.returncode != 0:
            detail = redact_sensitive_text(completed.stderr.strip())[-500:]
            raise ProviderError(
                f"Codex CLI exited with {completed.returncode}: {detail}"
            )
        return parse_codex_text_jsonl(completed.stdout)


def parse_codex_jsonl(
    raw_output: str,
    *,
    output_min: float = float("-inf"),
    output_max: float = float("inf"),
) -> PredictionResponse:
    completion = parse_codex_text_jsonl(raw_output)
    value = parse_numeric_prediction(completion.text)
    if not output_min <= value <= output_max:
        raise ProviderError(
            f"Prediction {value} is outside [{output_min}, {output_max}]"
        )
    return PredictionResponse(
        value=value,
        text=completion.text,
        usage=completion.usage,
        raw_output=completion.raw_output,
    )


def parse_codex_text_jsonl(raw_output: str) -> CompletionResponse:
    messages: list[str] = []
    usage = TokenUsage()
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                messages.append(str(item["text"]))
        elif event.get("type") == "turn.completed":
            payload = event.get("usage", {})
            usage = TokenUsage(
                input_tokens=int(payload.get("input_tokens", 0)),
                output_tokens=int(payload.get("output_tokens", 0)),
            )

    if not messages:
        raise ProviderError("Codex JSONL contained no agent_message event")
    text = messages[-1].strip()
    return CompletionResponse(
        text=text,
        usage=usage,
        raw_output=redact_sensitive_text(raw_output),
    )


def parse_numeric_prediction(text: str) -> float:
    stripped = text.strip().strip("`").strip()
    try:
        return float(stripped)
    except ValueError:
        matches = re.findall(
            r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
            stripped,
        )
        if len(matches) != 1:
            raise ProviderError(
                f"Expected one numeric prediction, found {len(matches)}"
            )
        return float(matches[0])


def _find_codex() -> str:
    executable = shutil.which("codex") or shutil.which("codex.cmd")
    if executable is None:
        raise FileNotFoundError(
            "Codex CLI was not found on PATH; pass --codex-executable"
        )
    return executable


def _safe_codex_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name.upper() in _SAFE_ENVIRONMENT_NAMES
    }
