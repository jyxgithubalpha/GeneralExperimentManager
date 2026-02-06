"""
MLPParamSpace - MLP 参数搜索空间
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseParamSpace


@dataclass
class MLPParamSpace(BaseParamSpace):
    """MLP 参数搜索空间"""
    learning_rate: Tuple[float, float] = (1e-4, 1e-2)
    batch_size_choices: List[int] = None
    dropout: Tuple[float, float] = (0.0, 0.5)
    hidden_dim_1: Tuple[int, int] = (64, 512)
    hidden_dim_2: Tuple[int, int] = (32, 256)
    hidden_dim_3: Tuple[int, int] = (16, 128)
    
    def __post_init__(self):
        if self.batch_size_choices is None:
            self.batch_size_choices = [256, 512, 1024, 2048]
    
    def sample(self, trial, shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        从搜索空间采样
        
        Args:
            trial: Optuna trial
            shrunk_space: 收缩后的搜索空间 (可选)
        """
        space = shrunk_space or {}
        
        lr_range = space.get("learning_rate", self.learning_rate)
        dropout_range = space.get("dropout", self.dropout)
        h1_range = space.get("hidden_dim_1", self.hidden_dim_1)
        h2_range = space.get("hidden_dim_2", self.hidden_dim_2)
        h3_range = space.get("hidden_dim_3", self.hidden_dim_3)
        
        hidden_dim_1 = trial.suggest_int("hidden_dim_1", int(h1_range[0]), int(h1_range[1]))
        hidden_dim_2 = trial.suggest_int("hidden_dim_2", int(h2_range[0]), int(h2_range[1]))
        hidden_dim_3 = trial.suggest_int("hidden_dim_3", int(h3_range[0]), int(h3_range[1]))
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr_range, log=True),
            "batch_size": trial.suggest_categorical("batch_size", self.batch_size_choices),
            "dropout": trial.suggest_float("dropout", *dropout_range),
            "hidden_dims": [hidden_dim_1, hidden_dim_2, hidden_dim_3],
        }
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """转换为字典格式"""
        return {
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "hidden_dim_1": self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
            "hidden_dim_3": self.hidden_dim_3,
        }
