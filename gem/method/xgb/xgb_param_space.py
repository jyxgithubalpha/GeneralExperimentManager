"""
XGBoost parameter search space.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseParamSpace


@dataclass
class XGBoostParamSpace(BaseParamSpace):
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    max_depth: Tuple[int, int] = (3, 12)
    min_child_weight: Tuple[float, float] = (1.0, 20.0)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    reg_alpha: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda: Tuple[float, float] = (1e-8, 10.0)
    gamma: Tuple[float, float] = (0.0, 5.0)

    def sample(self, trial, shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        lr_range = space.get("learning_rate", self.learning_rate)
        md_range = space.get("max_depth", self.max_depth)
        mcw_range = space.get("min_child_weight", self.min_child_weight)
        ss_range = space.get("subsample", self.subsample)
        cs_range = space.get("colsample_bytree", self.colsample_bytree)
        ra_range = space.get("reg_alpha", self.reg_alpha)
        rl_range = space.get("reg_lambda", self.reg_lambda)
        g_range = space.get("gamma", self.gamma)

        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr_range, log=True),
            "max_depth": trial.suggest_int("max_depth", int(md_range[0]), int(md_range[1])),
            "min_child_weight": trial.suggest_float("min_child_weight", *mcw_range),
            "subsample": trial.suggest_float("subsample", *ss_range),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cs_range),
            "reg_alpha": trial.suggest_float("reg_alpha", *ra_range, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *rl_range, log=True),
            "gamma": trial.suggest_float("gamma", *g_range),
        }

    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
        }

    def get_param_names(self) -> list:
        return [
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "gamma",
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
        md_range = space.get("max_depth", self.max_depth)
        mcw_range = space.get("min_child_weight", self.min_child_weight)
        ss_range = space.get("subsample", self.subsample)
        cs_range = space.get("colsample_bytree", self.colsample_bytree)
        ra_range = space.get("reg_alpha", self.reg_alpha)
        rl_range = space.get("reg_lambda", self.reg_lambda)
        g_range = space.get("gamma", self.gamma)

        return {
            "learning_rate": tune.loguniform(*lr_range),
            "max_depth": tune.randint(int(md_range[0]), int(md_range[1]) + 1),
            "min_child_weight": tune.uniform(*mcw_range),
            "subsample": tune.uniform(*ss_range),
            "colsample_bytree": tune.uniform(*cs_range),
            "reg_alpha": tune.loguniform(*ra_range),
            "reg_lambda": tune.loguniform(*rl_range),
            "gamma": tune.uniform(*g_range),
        }
