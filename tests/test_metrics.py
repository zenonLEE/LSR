import json
from pathlib import Path

import numpy as np
import pandas as pd

from lsr_co2.metrics import (
    compute_metrics,
    metrics_from_csv,
    sha256_file,
    verify_manifest,
)


def test_metrics_from_local_csv(tmp_path: Path) -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([1.1, 1.8, 3.2, 3.9])
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"true": true, "prediction": prediction}).to_csv(
        path, index=False
    )
    assert metrics_from_csv(path, "prediction") == compute_metrics(true, prediction)


def test_private_manifest_verification(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"true": [1.0, 2.0, 3.0], "prediction": [0.9, 2.1, 3.2]}
    ).to_csv(path, index=False)
    metrics = metrics_from_csv(path, "prediction")
    manifest = {
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "prediction_column": "prediction",
                "metrics": metrics.to_dict(),
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(manifest_path) == []

    path.write_text("true,prediction\n1,9\n", encoding="utf-8")
    assert verify_manifest(manifest_path) == [
        "predictions.csv: SHA256 mismatch"
    ]
