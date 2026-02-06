"""
CatBoostParamSpace - CatBoost 参数搜索空间
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseParamSpace


@dataclass
class CatBoostParamSpace(BaseParamSpace):
    """CatBoost 参数搜索空间"""
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    depth: Tuple[int, int] = (4, 10)
    l2_leaf_reg: Tuple[float, float] = (1.0, 10.0)
    bagging_temperature: Tuple[float, float] = (0.0, 1.0)
    random_strength: Tuple[float, float] = (0.0, 10.0)
    border_count: Tuple[int, int] = (32, 255)
    
    def sample(self, trial, shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        从搜索空间采样
        
        Args:
            trial: Optuna trial
            shrunk_space: 收缩后的搜索空间 (可选)
        """
        space = shrunk_space or {}
        
        lr_range = space.get("learning_rate", self.learning_rate)
        depth_range = space.get("depth", self.depth)
        l2_range = space.get("l2_leaf_reg", self.l2_leaf_reg)
        bt_range = space.get("bagging_temperature", self.bagging_temperature)
        rs_range = space.get("random_strength", self.random_strength)
        bc_range = space.get("border_count", self.border_count)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr_range, log=True),
            "depth": trial.suggest_int("depth", int(depth_range[0]), int(depth_range[1])),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *l2_range),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *bt_range),
            "random_strength": trial.suggest_float("random_strength", *rs_range),
            "border_count": trial.suggest_int("border_count", int(bc_range[0]), int(bc_range[1])),
        }
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """转换为字典格式"""
        return {
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "bagging_temperature": self.bagging_temperature,
            "random_strength": self.random_strength,
            "border_count": self.border_count,
        }
