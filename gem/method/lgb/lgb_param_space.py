"""
LightGBMParamSpace - LightGBM 参数搜索空间
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseParamSpace


@dataclass
class LightGBMParamSpace(BaseParamSpace):
    """LightGBM 参数搜索空间"""
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    num_leaves: Tuple[int, int] = (20, 300)
    max_depth: Tuple[int, int] = (3, 12)
    min_child_samples: Tuple[int, int] = (5, 100)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    reg_alpha: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda: Tuple[float, float] = (1e-8, 10.0)
    
    def sample(self, trial, shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        从搜索空间采样
        
        Args:
            trial: Optuna trial
            shrunk_space: 收缩后的搜索空间 (可选)
        """
        space = shrunk_space or {}
        
        lr_range = space.get("learning_rate", self.learning_rate)
        nl_range = space.get("num_leaves", self.num_leaves)
        md_range = space.get("max_depth", self.max_depth)
        mcs_range = space.get("min_child_samples", self.min_child_samples)
        ss_range = space.get("subsample", self.subsample)
        cs_range = space.get("colsample_bytree", self.colsample_bytree)
        ra_range = space.get("reg_alpha", self.reg_alpha)
        rl_range = space.get("reg_lambda", self.reg_lambda)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr_range, log=True),
            "num_leaves": trial.suggest_int("num_leaves", int(nl_range[0]), int(nl_range[1])),
            "max_depth": trial.suggest_int("max_depth", int(md_range[0]), int(md_range[1])),
            "min_child_samples": trial.suggest_int("min_child_samples", int(mcs_range[0]), int(mcs_range[1])),
            "subsample": trial.suggest_float("subsample", *ss_range),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cs_range),
            "reg_alpha": trial.suggest_float("reg_alpha", *ra_range, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *rl_range, log=True),
        }
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """转换为字典格式"""
        return {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }
