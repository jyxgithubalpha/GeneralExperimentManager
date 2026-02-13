"""
CatBoost parameter search space.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseParamSpace


@dataclass
class CatBoostParamSpace(BaseParamSpace):
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    depth: Tuple[int, int] = (4, 10)
    l2_leaf_reg: Tuple[float, float] = (1e-3, 10.0)
    bagging_temperature: Tuple[float, float] = (0.0, 1.0)
    random_strength: Tuple[float, float] = (0.0, 1.0)
    rsm: Tuple[float, float] = (0.5, 1.0)

    def sample(self, trial, shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        lr_range = space.get("learning_rate", self.learning_rate)
        depth_range = space.get("depth", self.depth)
        l2_range = space.get("l2_leaf_reg", self.l2_leaf_reg)
        bt_range = space.get("bagging_temperature", self.bagging_temperature)
        rs_range = space.get("random_strength", self.random_strength)
        rsm_range = space.get("rsm", self.rsm)

        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr_range, log=True),
            "depth": trial.suggest_int("depth", int(depth_range[0]), int(depth_range[1])),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *l2_range, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *bt_range),
            "random_strength": trial.suggest_float("random_strength", *rs_range),
            "rsm": trial.suggest_float("rsm", *rsm_range),
        }

    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        return {
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "bagging_temperature": self.bagging_temperature,
            "random_strength": self.random_strength,
            "rsm": self.rsm,
        }

    def get_param_names(self) -> list:
        return [
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "bagging_temperature",
            "random_strength",
            "rsm",
        ]

    def to_ray_tune_space(
        self,
        shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        try:
            from ray import tune
        except ImportError:
            raise ImportError("ray[tune] is required. Install with: pip install 'ray[tune]'")

        space = shrunk_space or {}
        lr_range = space.get("learning_rate", self.learning_rate)
        depth_range = space.get("depth", self.depth)
        l2_range = space.get("l2_leaf_reg", self.l2_leaf_reg)
        bt_range = space.get("bagging_temperature", self.bagging_temperature)
        rs_range = space.get("random_strength", self.random_strength)
        rsm_range = space.get("rsm", self.rsm)

        return {
            "learning_rate": tune.loguniform(*lr_range),
            "depth": tune.randint(int(depth_range[0]), int(depth_range[1]) + 1),
            "l2_leaf_reg": tune.loguniform(*l2_range),
            "bagging_temperature": tune.uniform(*bt_range),
            "random_strength": tune.uniform(*rs_range),
            "rsm": tune.uniform(*rsm_range),
        }
