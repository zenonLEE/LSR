from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NeighborQuery:
    indices: np.ndarray
    distances: np.ndarray


class LocalSubspaceRetriever:
    """Standardize numeric features and retrieve nearest training samples."""

    def __init__(self, feature_columns: tuple[str, ...], n_neighbors: int) -> None:
        self.feature_columns = feature_columns
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.index = NearestNeighbors(n_neighbors=n_neighbors)
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> LocalSubspaceRetriever:
        features = train.loc[:, self.feature_columns].to_numpy(dtype=float)
        if not np.isfinite(features).all():
            raise ValueError("Training features contain missing or non-finite values")
        self.index.fit(self.scaler.fit_transform(features))
        self._fitted = True
        return self

    def query(self, samples: pd.DataFrame) -> NeighborQuery:
        if not self._fitted:
            raise RuntimeError("Retriever must be fitted before query")
        features = samples.loc[:, self.feature_columns].to_numpy(dtype=float)
        if not np.isfinite(features).all():
            raise ValueError("Query features contain missing or non-finite values")
        distances, indices = self.index.kneighbors(self.scaler.transform(features))
        return NeighborQuery(indices=indices, distances=distances)


def multi_scale_statistics(
    neighbor_targets: np.ndarray,
    scales: tuple[int, ...],
    *,
    short_std_ratio: float,
    long_std_threshold: float,
) -> dict[str, object]:
    values = np.asarray(neighbor_targets, dtype=float)
    if values.ndim != 1 or len(values) < scales[-1]:
        raise ValueError("neighbor_targets must cover the largest scale")

    output: dict[str, object] = {}
    for scale in scales:
        window = values[:scale]
        output[f"nb{scale}"] = {
            "k": scale,
            "mean": round(float(window.mean()), 3),
            "median": round(float(np.median(window)), 3),
            "std": round(float(window.std(ddof=0)), 3),
            "min": round(float(window.min()), 3),
            "max": round(float(window.max()), 3),
        }

    short = output[f"nb{scales[0]}"]
    long = output[f"nb{scales[-1]}"]
    assert isinstance(short, dict) and isinstance(long, dict)
    output["scale_consistency"] = {
        "short_vs_long_diff": round(
            abs(float(short["mean"]) - float(long["mean"])), 3
        ),
        "short_window_tighter": bool(
            float(short["std"]) < float(long["std"]) * short_std_ratio
        ),
        "long_window_unstable": bool(
            float(long["std"]) > long_std_threshold
        ),
    }
    return output
