"""
XGBoostParamSpace - XGBoost 参数搜索空间
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseParamSpace


@dataclass
class XGBoostParamSpace(BaseParamSpace):
    """XGBoost 参数搜索空间"""
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    max_depth: Tuple[int, int] = (3, 12)
    min_child_weight: Tuple[int, int] = (1, 10)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    gamma: Tuple[float, float] = (0.0, 5.0)
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
        md_range = space.get("max_depth", self.max_depth)
        mcw_range = space.get("min_child_weight", self.min_child_weight)
        ss_range = space.get("subsample", self.subsample)
        cs_range = space.get("colsample_bytree", self.colsample_bytree)
        gm_range = space.get("gamma", self.gamma)
        ra_range = space.get("reg_alpha", self.reg_alpha)
        rl_range = space.get("reg_lambda", self.reg_lambda)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr_range, log=True),
            "max_depth": trial.suggest_int("max_depth", int(md_range[0]), int(md_range[1])),
            "min_child_weight": trial.suggest_int("min_child_weight", int(mcw_range[0]), int(mcw_range[1])),
            "subsample": trial.suggest_float("subsample", *ss_range),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cs_range),
            "gamma": trial.suggest_float("gamma", *gm_range),
            "reg_alpha": trial.suggest_float("reg_alpha", *ra_range, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *rl_range, log=True),
        }
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """转换为字典格式"""
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }
