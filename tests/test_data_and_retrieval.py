import numpy as np

from lsr_co2.config import ExperimentConfig, RefinementConfig
from lsr_co2.datasets import load_dataset, split_dataset
from lsr_co2.retrieval import LocalSubspaceRetriever, multi_scale_statistics


def test_biochar_random_holdout_uses_synthetic_data(
    biochar_config: ExperimentConfig,
) -> None:
    frame = load_dataset(biochar_config)
    split = split_dataset(frame, biochar_config)
    assert frame.shape == (120, 23)
    assert (len(split.train), len(split.test)) == (96, 24)
    assert frame[list(biochar_config.feature_columns)].isna().sum().sum() == 0


def test_dac_locked_split_and_retrieval(
    dac_configs: tuple[ExperimentConfig, RefinementConfig],
) -> None:
    config, _ = dac_configs
    frame = load_dataset(config)
    split = split_dataset(frame, config)
    assert (len(split.train), len(split.test)) == (117, 30)
    query = LocalSubspaceRetriever(config.feature_columns, config.k_max).fit(
        split.train
    ).query(split.test.iloc[:2])
    assert query.indices.shape == (2, 15)
    assert np.all(np.diff(query.distances, axis=1) >= 0)


def test_multiscale_statistics() -> None:
    stats = multi_scale_statistics(
        np.arange(1, 16, dtype=float),
        (7, 11, 15),
        short_std_ratio=0.7,
        long_std_threshold=0.6,
    )
    assert stats["nb7"]["mean"] == 4.0
    assert stats["nb15"]["median"] == 8.0
    assert stats["scale_consistency"]["short_vs_long_diff"] == 4.0
