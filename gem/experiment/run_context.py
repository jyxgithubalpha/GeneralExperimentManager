"""
Run-time context shared by manager, executor and split runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ..method.method_dataclasses import TrainConfig
from .experiment_dataclasses import ExperimentConfig, StatePolicyConfig


@dataclass(frozen=True)
class RunContext:
    experiment_config: ExperimentConfig
    train_config: TrainConfig
    method_config: Optional[Mapping[str, Any]] = None
    transform_config: Optional[Any] = None
    adapter_config: Optional[Any] = None
    output_dir: Path = Path(".")
    seed: int = 42

    @property
    def state_policy(self) -> StatePolicyConfig:
        return self.experiment_config.state_policy

    @property
    def n_trials(self) -> int:
        return self.experiment_config.n_trials

    @property
    def parallel_trials(self) -> int:
        return self.experiment_config.parallel_trials

    @property
    def trial_timeout(self) -> Optional[int]:
        return self.experiment_config.trial_timeout

    @property
    def use_ray_tune(self) -> bool:
        if self.experiment_config.use_ray_tune is not None:
            return bool(self.experiment_config.use_ray_tune)
        return self.parallel_trials > 1

    @property
    def do_tune(self) -> bool:
        return self.n_trials > 0

    def split_dir(self, split_id: int) -> Path:
        path = self.output_dir / f"split_{split_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def trial_dir(self, split_id: int, trial_id: int) -> Path:
        path = self.split_dir(split_id) / f"trial_{trial_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path
