"""
Runtime bootstrap utilities for Hydra configuration instantiation.
"""


from dataclasses import dataclass
from typing import Any, Mapping

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .data import DataModule, SplitGenerator
from .experiment import ExperimentConfig
from .method.method_dataclasses import TrainConfig


@dataclass(frozen=True)
class RuntimeBundle:
    split_generator: SplitGenerator
    data_module: DataModule
    experiment_config: ExperimentConfig
    train_config: TrainConfig
    method_config: Mapping[str, Any]


def _require_config_nodes(cfg: DictConfig, paths: list[str]) -> None:
    missing = [path for path in paths if OmegaConf.select(cfg, path, default=None) is None]
    if missing:
        raise ValueError(f"Missing required Hydra config nodes: {missing}")


def _require_seed(cfg: DictConfig) -> int:
    seed = OmegaConf.select(cfg, "seed", default=None)
    if seed is None:
        raise ValueError("Missing required root config value: seed")
    return int(seed)


def _require_non_empty_cfg_value(cfg: DictConfig, path: str) -> None:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise ValueError(f"Missing required Hydra config value: {path}")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Hydra config value must not be empty: {path}")


def _validate_runtime_cfg(cfg: DictConfig) -> None:
    _require_non_empty_cfg_value(cfg, "method.name")
    _require_config_nodes(
        cfg,
        [
            "method.trainer",
            "method.evaluator",
            "method.importance_extractor",
        ],
    )

    n_trials = int(OmegaConf.select(cfg, "experiment.n_trials", default=0) or 0)
    tuner_cfg = OmegaConf.select(cfg, "method.tuner", default=None)
    if n_trials > 0 and tuner_cfg is None:
        raise ValueError(
            "Invalid runtime config: experiment.n_trials > 0 requires method.tuner."
        )


def build_runtime(cfg: DictConfig) -> RuntimeBundle:
    _require_config_nodes(
        cfg,
        [
            "splitgenerator",
            "datamodule",
            "method",
            "method.train_config",
            "experiment",
        ],
    )
    _validate_runtime_cfg(cfg)
    seed = _require_seed(cfg)

    split_generator = instantiate(cfg.splitgenerator)
    data_module = instantiate(cfg.datamodule)
    experiment_config = instantiate(cfg.experiment, seed=seed)
    train_config = instantiate(cfg.method.train_config, seed=seed)

    return RuntimeBundle(
        split_generator=split_generator,
        data_module=data_module,
        experiment_config=experiment_config,
        train_config=train_config,
        method_config=cfg.method,
    )
