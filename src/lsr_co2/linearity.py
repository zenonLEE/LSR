from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import LinearityConfig
from .datasets import BIOCHAR_COLUMN_NAMES
from .metrics import sha256_file


@dataclass(frozen=True)
class LinearitySummary:
    experiment: str
    output_dir: str
    train_size: int
    test_size: int
    feature_count: int
    neighbor_count: int
    mean_support_fit_r2: float
    mean_local_linear_r2: float
    mean_local_nonlinear_r2: float
    local_linear_to_nonlinear_ratio: float
    relative_gain_min_pct: float
    relative_gain_max_pct: float


@dataclass(frozen=True)
class LinearityRun:
    support_statistics: pd.DataFrame
    model_comparison: pd.DataFrame
    k_sensitivity: pd.DataFrame
    local_scores: dict[str, np.ndarray]
    summary: LinearitySummary


def load_linearity_frame(config: LinearityConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Linearity dataset not found: {config.data_path}")
    frame = pd.read_csv(config.data_path).rename(columns=BIOCHAR_COLUMN_NAMES)
    required = set(config.feature_columns) | set(config.target_columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Linearity dataset is missing columns: {missing}")
    numeric = frame.loc[:, [*config.feature_columns, *config.target_columns]]
    non_numeric = [
        column for column in numeric
        if not pd.api.types.is_numeric_dtype(numeric[column])
    ]
    if non_numeric:
        raise TypeError(f"Expected numeric linearity columns: {non_numeric}")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Linearity inputs contain missing or non-finite values")
    return frame


def split_linearity_indices(
    row_count: int,
    config: LinearityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(row_count)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.split_seed,
    )
    return np.asarray(train_indices), np.asarray(test_indices)


def local_support_fit_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    neighbor_count: int,
    ridge_alpha: float,
) -> np.ndarray:
    """Return in-sample Ridge R2 on each query-selected training support."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.asarray(x_train, dtype=float))
    query_scaled = scaler.transform(np.asarray(x_query, dtype=float))
    index = NearestNeighbors(n_neighbors=neighbor_count).fit(train_scaled)
    neighbor_indices = index.kneighbors(query_scaled, return_distance=False)
    target = np.asarray(y_train, dtype=float)
    scores: list[float] = []
    for indices in neighbor_indices:
        local_x = train_scaled[indices]
        local_y = target[indices]
        model = Ridge(alpha=ridge_alpha).fit(local_x, local_y)
        scores.append(float(model.score(local_x, local_y)))
    return np.asarray(scores)


def compare_local_and_global_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: LinearityConfig,
) -> dict[str, float]:
    """Evaluate historical global/local linear and nonlinear query predictors."""
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "Local-linearity comparison requires: pip install -e '.[analysis]'"
        ) from error

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.asarray(x_train, dtype=float))
    test_scaled = scaler.transform(np.asarray(x_test, dtype=float))
    train_target = np.asarray(y_train, dtype=float)
    test_target = np.asarray(y_test, dtype=float)

    global_linear = LinearRegression().fit(train_scaled, train_target)
    global_linear_r2 = float(global_linear.score(test_scaled, test_target))
    global_nonlinear = XGBRegressor(
        n_estimators=config.global_xgb_estimators,
        max_depth=config.global_xgb_depth,
        random_state=config.model_seed,
        verbosity=0,
    ).fit(train_scaled, train_target)
    global_nonlinear_r2 = float(global_nonlinear.score(test_scaled, test_target))

    index = NearestNeighbors(n_neighbors=config.neighbor_count).fit(train_scaled)
    neighbor_indices = index.kneighbors(test_scaled, return_distance=False)
    local_linear_predictions: list[float] = []
    local_nonlinear_predictions: list[float] = []
    for query, indices in zip(test_scaled, neighbor_indices, strict=True):
        local_x = train_scaled[indices]
        local_y = train_target[indices]
        local_linear = Ridge(alpha=config.ridge_alpha).fit(local_x, local_y)
        local_linear_predictions.append(float(local_linear.predict([query])[0]))
        local_nonlinear = XGBRegressor(
            n_estimators=config.local_xgb_estimators,
            max_depth=config.local_xgb_depth,
            random_state=config.model_seed,
            verbosity=0,
        ).fit(local_x, local_y)
        local_nonlinear_predictions.append(float(local_nonlinear.predict([query])[0]))

    local_linear_r2 = float(r2_score(test_target, local_linear_predictions))
    local_nonlinear_r2 = float(r2_score(test_target, local_nonlinear_predictions))
    absolute_gain = local_linear_r2 - global_linear_r2
    relative_gain_pct = absolute_gain / global_linear_r2 * 100.0
    return {
        "Global_Linear": global_linear_r2,
        "Global_Nonlinear": global_nonlinear_r2,
        "Local_Linear": local_linear_r2,
        "Local_Nonlinear": local_nonlinear_r2,
        "Linearity_Gain": absolute_gain,
        "Relative_Gain_Pct": relative_gain_pct,
        "Local_Linear_vs_Nonlinear": local_linear_r2 / local_nonlinear_r2,
    }


def run_linearity_analysis(
    config: LinearityConfig,
    output_dir: str | Path,
    *,
    save_figure: bool = True,
) -> LinearityRun:
    frame = load_linearity_frame(config)
    train_indices, test_indices = split_linearity_indices(len(frame), config)
    features = frame.loc[:, config.feature_columns].to_numpy(dtype=float)
    x_train = features[train_indices]
    x_test = features[test_indices]
    support_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    local_scores: dict[str, np.ndarray] = {}

    for label, column in config.targets:
        target = frame[column].to_numpy(dtype=float)
        y_train = target[train_indices]
        y_test = target[test_indices]
        scores = local_support_fit_scores(
            x_train,
            y_train,
            x_test,
            neighbor_count=config.neighbor_count,
            ridge_alpha=config.ridge_alpha,
        )
        local_scores[label] = scores
        support_rows.append({
            "Target": label,
            "Local_R2_Mean": float(scores.mean()),
            "Local_R2_Std": float(scores.std(ddof=0)),
            "Local_R2_Min": float(scores.min()),
            "Local_R2_Max": float(scores.max()),
        })
        comparison_rows.append({
            "Target": label,
            **compare_local_and_global_models(
                x_train, y_train, x_test, y_test, config
            ),
        })

    support = pd.DataFrame(support_rows)
    comparison = pd.DataFrame(comparison_rows)
    sensitivity_column = config.sensitivity_target
    sensitivity_target = frame[sensitivity_column].to_numpy(dtype=float)[train_indices]
    sensitivity_rows: list[dict[str, float | int]] = []
    for neighbor_count in config.sensitivity_k:
        scores = local_support_fit_scores(
            x_train,
            sensitivity_target,
            x_test,
            neighbor_count=neighbor_count,
            ridge_alpha=config.ridge_alpha,
        )
        sensitivity_rows.append({
            "k": neighbor_count,
            "mean_r2": float(scores.mean()),
            "std_r2": float(scores.std(ddof=0)),
            "min_r2": float(scores.min()),
            "max_r2": float(scores.max()),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    local_linear_mean = float(comparison["Local_Linear"].mean())
    local_nonlinear_mean = float(comparison["Local_Nonlinear"].mean())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = LinearitySummary(
        experiment=config.name,
        output_dir=str(output.resolve()),
        train_size=len(train_indices),
        test_size=len(test_indices),
        feature_count=len(config.feature_columns),
        neighbor_count=config.neighbor_count,
        mean_support_fit_r2=float(support["Local_R2_Mean"].mean()),
        mean_local_linear_r2=local_linear_mean,
        mean_local_nonlinear_r2=local_nonlinear_mean,
        local_linear_to_nonlinear_ratio=local_linear_mean / local_nonlinear_mean,
        relative_gain_min_pct=float(comparison["Relative_Gain_Pct"].min()),
        relative_gain_max_pct=float(comparison["Relative_Gain_Pct"].max()),
    )
    support.to_csv(output / "local_linearity_stats.csv", index=False)
    comparison.to_csv(output / "local_vs_global_comparison.csv", index=False)
    sensitivity.to_csv(output / "k_sensitivity_analysis.csv", index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "data_sha256": sha256_file(config.data_path),
        "support_fit_definition": (
            "In-sample Ridge R2 on each query-selected training neighborhood."
        ),
        "comparison_definition": (
            "R2 on the fixed held-out query set; query targets are not used in fitting."
        ),
        "relative_gain_formula": (
            "100 * (Local_Linear - Global_Linear) / Global_Linear"
        ),
        "summary": asdict(summary),
    }
    (output / "linearity.metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    if save_figure:
        create_linearity_figure(
            comparison,
            sensitivity,
            local_scores,
            output / "local_linearity_verification.png",
        )
    return LinearityRun(
        support_statistics=support,
        model_comparison=comparison,
        k_sensitivity=sensitivity,
        local_scores=local_scores,
        summary=summary,
    )


def create_linearity_figure(
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    local_scores: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Linearity figures require: pip install -e '.[analysis]'"
        ) from error

    targets = comparison["Target"].tolist()
    x = np.arange(len(targets))
    width = 0.2
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(
        x - 1.5 * width, comparison["Global_Linear"], width, label="Global linear"
    )
    axes[0, 0].bar(
        x - 0.5 * width,
        comparison["Global_Nonlinear"],
        width,
        label="Global nonlinear",
    )
    axes[0, 0].bar(
        x + 0.5 * width, comparison["Local_Linear"], width, label="Local linear"
    )
    axes[0, 0].bar(
        x + 1.5 * width,
        comparison["Local_Nonlinear"],
        width,
        label="Local nonlinear",
    )
    axes[0, 0].set_xticks(x, targets)
    axes[0, 0].set_ylabel("Held-out R2")
    axes[0, 0].set_title("Global and query-centred predictors")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    for target, scores in local_scores.items():
        axes[0, 1].hist(scores, bins=20, alpha=0.5, label=target)
    axes[0, 1].set_xlabel("In-support Ridge R2")
    axes[0, 1].set_ylabel("Query count")
    axes[0, 1].set_title("Local support fit at k=30")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].errorbar(
        sensitivity["k"],
        sensitivity["mean_r2"],
        yerr=sensitivity["std_r2"],
        marker="o",
        capsize=5,
    )
    axes[1, 0].set_xlabel("Neighbors (k)")
    axes[1, 0].set_ylabel("In-support Ridge R2")
    axes[1, 0].set_title("CO2 uptake support-fit sensitivity")
    axes[1, 0].grid(alpha=0.3)

    bars = axes[1, 1].bar(targets, comparison["Relative_Gain_Pct"])
    axes[1, 1].set_ylabel("Relative held-out R2 gain (%)")
    axes[1, 1].set_title("Local linear relative to global linear")
    axes[1, 1].grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, comparison["Relative_Gain_Pct"], strict=True):
        axes[1, 1].annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
