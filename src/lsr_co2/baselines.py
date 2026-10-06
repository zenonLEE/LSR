from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig

BIOCHAR_XGBOOST_PARAMS = {
    "n_estimators": 675,
    "max_depth": 5,
    "learning_rate": 0.045,
    "subsample": 0.87,
    "colsample_bytree": 0.75,
    "min_child_weight": 6.2,
    "reg_lambda": 0.15,
    "reg_alpha": 0.73,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

DAC_REPEATED_SPLIT_XGBOOST_PARAMS = {
    "n_estimators": 800,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.80,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def xgb_reference_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ExperimentConfig,
) -> np.ndarray:
    if not config.xgb_reference.startswith("fit"):
        return test[config.xgb_reference].to_numpy(dtype=float)

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "Live fitted baselines require: pip install -e '.[live]'"
        ) from error

    parameters = (
        DAC_REPEATED_SPLIT_XGBOOST_PARAMS
        if config.xgb_reference == "fit_dac_repeated"
        else BIOCHAR_XGBOOST_PARAMS
    )
    model = XGBRegressor(**parameters)
    model.fit(
        train.loc[:, config.feature_columns].to_numpy(dtype=float),
        train[config.target_column].to_numpy(dtype=float),
    )
    return np.asarray(
        model.predict(
            test.loc[:, config.feature_columns].to_numpy(dtype=float)
        ),
        dtype=float,
    )
