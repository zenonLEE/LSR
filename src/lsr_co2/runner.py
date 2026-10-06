from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import xgb_reference_predictions
from .config import ExperimentConfig
from .datasets import load_dataset, split_dataset
from .metrics import RegressionMetrics, compute_metrics
from .prompts import build_prompt
from .providers import PredictionProvider, ProviderError
from .retrieval import LocalSubspaceRetriever
from .security import redact_sensitive_text

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    experiment: str
    output_path: str
    completed: int
    fallback_count: int
    input_tokens: int
    output_tokens: int
    metrics: RegressionMetrics


def run_experiment(
    config: ExperimentConfig,
    provider: PredictionProvider,
    output_path: str | Path,
    *,
    model_name: str,
    resume: bool = False,
    allow_fallback: bool = False,
    limit: int | None = None,
    save_raw: bool = False,
) -> RunSummary:
    frame = load_dataset(config)
    split = split_dataset(frame, config)
    rules_text = config.rules_path.read_text(encoding="utf-8")
    xgb_predictions = xgb_reference_predictions(
        split.train, split.test, config
    )

    retriever = LocalSubspaceRetriever(
        config.feature_columns, config.k_max
    ).fit(split.train)
    query = retriever.query(split.test)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_progress(output) if resume else []
    completed_ids = {int(row["sample_id"]) for row in rows}
    raw_path = output.with_suffix(".responses.jsonl")
    token_in = token_out = 0

    stop = len(split.test) if limit is None else min(
        limit, len(split.test)
    )
    for sample_id in range(stop):
        if sample_id in completed_ids:
            continue
        test_row = split.test.iloc[sample_id]
        neighbor_rows = [
            split.train.iloc[index]
            for index in query.indices[sample_id]
        ]
        prompt = build_prompt(
            config,
            test_row,
            neighbor_rows,
            float(xgb_predictions[sample_id]),
            rules_text,
            model_name=model_name,
        )
        status = "llm"
        error_message = ""
        raw_output = ""
        try:
            response = provider.predict(prompt)
            prediction = response.value
            raw_output = response.raw_output
            token_in += response.usage.input_tokens
            token_out += response.usage.output_tokens
        except ProviderError as error:
            if not allow_fallback:
                _write_progress(output, rows)
                raise
            prediction = float(np.mean([
                row[config.target_column] for row in neighbor_rows[:5]
            ]))
            status = "fallback"
            error_message = str(error)

        truth = float(test_row[config.target_column])
        rows.append({
            "sample_id": sample_id,
            "true": truth,
            "xgb_pred": float(xgb_predictions[sample_id]),
            config.prediction_column: float(prediction),
            "error": float(prediction - truth),
            "status": status,
            "provider_error": error_message,
        })
        _write_progress(output, rows)
        if save_raw:
            _append_raw_response(raw_path, sample_id, raw_output)
        LOGGER.info(
            "[%d/%d] true=%.3f prediction=%.3f status=%s",
            sample_id + 1,
            stop,
            truth,
            prediction,
            status,
        )

    result_frame = pd.DataFrame(rows).sort_values("sample_id")
    if len(result_frame) != stop:
        raise RuntimeError(
            f"Run produced {len(result_frame)} rows; expected {stop}"
        )
    metrics = compute_metrics(
        result_frame["true"].to_numpy(dtype=float),
        result_frame[config.prediction_column].to_numpy(dtype=float),
    )
    summary = RunSummary(
        experiment=config.name,
        output_path=str(output.resolve()),
        completed=len(result_frame),
        fallback_count=int(
            (result_frame["status"] == "fallback").sum()
        ),
        input_tokens=token_in,
        output_tokens=token_out,
        metrics=metrics,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "split_seed": config.split_seed,
        "k_max": config.k_max,
        "scales": list(config.scales),
        **asdict(summary),
        "metrics": metrics.to_dict(),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return summary


def _load_progress(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _write_progress(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).sort_values("sample_id").to_csv(
        temporary, index=False
    )
    temporary.replace(path)


def _append_raw_response(
    path: Path,
    sample_id: int,
    raw_output: str,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "sample_id": sample_id,
            "raw_output": redact_sensitive_text(raw_output),
        }) + "\n")
