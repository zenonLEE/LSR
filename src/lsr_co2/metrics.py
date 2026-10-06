from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    r2: float
    mae: float
    bias: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with equal shape")
    if len(true) < 2 or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require at least two finite observations")
    residual = pred - true
    denominator = float(np.square(true - true.mean()).sum())
    if denominator == 0:
        raise ValueError("R2 is undefined for a constant target")
    return RegressionMetrics(
        n=len(true),
        r2=float(1.0 - np.square(residual).sum() / denominator),
        mae=float(np.abs(residual).mean()),
        bias=float(residual.mean()),
    )


def metrics_from_csv(
    path: str | Path,
    prediction_column: str,
    target_column: str = "true",
) -> RegressionMetrics:
    frame = pd.read_csv(path)
    missing = {target_column, prediction_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    return compute_metrics(
        frame[target_column].to_numpy(dtype=float),
        frame[prediction_column].to_numpy(dtype=float),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path: str | Path, tolerance: float = 1e-9) -> list[str]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    base_dir = (manifest_path.parent / manifest.get("base_dir", ".")).resolve()
    failures: list[str] = []

    for artifact in manifest["artifacts"]:
        artifact_path = base_dir / artifact["path"]
        label = artifact["path"]
        if not artifact_path.exists():
            failures.append(f"{label}: file is missing")
            continue
        if sha256_file(artifact_path) != artifact["sha256"]:
            failures.append(f"{label}: SHA256 mismatch")
            continue
        metrics = metrics_from_csv(
            artifact_path,
            artifact["prediction_column"],
            artifact.get("target_column", "true"),
        ).to_dict()
        for key, expected in artifact["metrics"].items():
            actual = metrics[key]
            if key == "n":
                if actual != expected:
                    failures.append(f"{label}: n={actual}, expected {expected}")
            elif abs(float(actual) - float(expected)) > tolerance:
                failures.append(f"{label}: {key}={actual}, expected {expected}")
    return failures
