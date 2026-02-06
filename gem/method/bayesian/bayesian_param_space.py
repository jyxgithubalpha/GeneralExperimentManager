"""
BayesianParamSpace - 贝叶斯模型参数搜索空间
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseParamSpace


@dataclass
class BayesianParamSpace(BaseParamSpace):
    """贝叶斯模型参数搜索空间"""
    model_type_choices: List[str] = None
    alpha_1: Tuple[float, float] = (1e-7, 1e-5)
    alpha_2: Tuple[float, float] = (1e-7, 1e-5)
    lambda_1: Tuple[float, float] = (1e-7, 1e-5)
    lambda_2: Tuple[float, float] = (1e-7, 1e-5)
    n_iter: Tuple[int, int] = (100, 500)
    tol: Tuple[float, float] = (1e-4, 1e-2)
    
    def __post_init__(self):
        if self.model_type_choices is None:
            self.model_type_choices = ["bayesian_ridge", "ard"]
    
    def sample(self, trial, shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        从搜索空间采样
        
        Args:
            trial: Optuna trial
            shrunk_space: 收缩后的搜索空间 (可选)
        """
        space = shrunk_space or {}
        
        a1_range = space.get("alpha_1", self.alpha_1)
        a2_range = space.get("alpha_2", self.alpha_2)
        l1_range = space.get("lambda_1", self.lambda_1)
        l2_range = space.get("lambda_2", self.lambda_2)
        n_iter_range = space.get("n_iter", self.n_iter)
        tol_range = space.get("tol", self.tol)
        
        return {
            "model_type": trial.suggest_categorical("model_type", self.model_type_choices),
            "alpha_1": trial.suggest_float("alpha_1", *a1_range, log=True),
            "alpha_2": trial.suggest_float("alpha_2", *a2_range, log=True),
            "lambda_1": trial.suggest_float("lambda_1", *l1_range, log=True),
            "lambda_2": trial.suggest_float("lambda_2", *l2_range, log=True),
            "n_iter": trial.suggest_int("n_iter", int(n_iter_range[0]), int(n_iter_range[1])),
            "tol": trial.suggest_float("tol", *tol_range, log=True),
        }
    
    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """转换为字典格式"""
        return {
            "alpha_1": self.alpha_1,
            "alpha_2": self.alpha_2,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "n_iter": self.n_iter,
            "tol": self.tol,
        }
