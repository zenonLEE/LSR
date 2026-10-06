from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig

BIOCHAR_COLUMN_NAMES = {
    "Plant-based structural composition (%) - Cellulose": "Cellulose_pct",
    "Plant-based structural composition (%) - Hemicellulose": "Hemicellulose_pct",
    "Plant-based structural composition (%) - Lignin": "Lignin_pct",
    "Elemental composition (%) - C": "C_pct",
    "Elemental composition (%) - H": "H_pct",
    "Elemental composition (%) - N": "N_pct",
    "Elemental composition (%) - S": "S_pct",
    "Elemental composition (%) - O": "O_pct",
    "Carbonization - Hydrothermal": "HTC",
    "Carbonization - Temperature": "Carb_Temp",
    "Carbonization - Ramping rate": "Carb_Ramp",
    "Carbonization - Time": "Carb_Time",
    "Activation - Agent": "Act_Agent",
    "Activation - Agent mass": "Act_Mass",
    "Activation - Temperature": "Act_Temp",
    "Activation - Ramping rate": "Act_Ramp",
    "Activation - Time/h": "Act_Time_h",
    "Adsorption condition - Temperature": "Ads_Temp",
    "Adsorption condition - CO2 partial pressure": "CO2_PP",
    "Textural properties - SBET (m2/g)": "SBET",
    "Textural properties - Vtotal (cm3/g)": "Vtotal",
    "Textural properties - Vnarrow (cm3/g)": "Vnarrow",
    "CO2 adsorption uptake": "CO2_uptake",
}


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def load_dataset(config: ExperimentConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.data_path}")
    frame = pd.read_csv(config.data_path)
    if config.dataset == "biochar":
        frame = frame.rename(columns=BIOCHAR_COLUMN_NAMES)
    _validate_frame(frame, config)
    if config.drop_missing:
        frame = frame.dropna(
            subset=[*config.feature_columns, config.target_column]
        ).reset_index(drop=True)
    return frame


def split_dataset(frame: pd.DataFrame, config: ExperimentConfig) -> DatasetSplit:
    if config.split_strategy == "random_holdout":
        indices = np.arange(len(frame))
        train_indices, test_indices = train_test_split(
            indices, test_size=config.test_size, random_state=config.split_seed
        )
        train = frame.iloc[train_indices].reset_index(drop=True)
        test = frame.iloc[test_indices].reset_index(drop=True)
    else:
        split_column = str(config.split_column)
        train = frame[frame[split_column] == config.train_value].reset_index(drop=True)
        test = frame[frame[split_column] == config.test_value].reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError("The configured split produced an empty partition")
    if len(train) < config.k_max:
        raise ValueError(f"k_max={config.k_max} exceeds train size={len(train)}")
    return DatasetSplit(train=train, test=test)


def dataset_summary(config: ExperimentConfig) -> dict[str, object]:
    frame = load_dataset(config)
    split = split_dataset(frame, config)
    return {
        "dataset": config.dataset,
        "path": str(config.data_path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "features": len(config.feature_columns),
        "missing_feature_values": int(frame[list(config.feature_columns)].isna().sum().sum()),
        "missing_targets": int(frame[config.target_column].isna().sum()),
    }


def _validate_frame(frame: pd.DataFrame, config: ExperimentConfig) -> None:
    required = set(config.feature_columns) | {config.target_column}
    if config.split_strategy == "locked_column":
        required.add(str(config.split_column))
    if not config.xgb_reference.startswith("fit"):
        required.add(config.xgb_reference)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    numeric = frame[list(config.feature_columns) + [config.target_column]]
    non_numeric = [
        column for column in numeric
        if not pd.api.types.is_numeric_dtype(numeric[column])
    ]
    if non_numeric:
        raise TypeError(f"Expected numeric columns: {non_numeric}")
