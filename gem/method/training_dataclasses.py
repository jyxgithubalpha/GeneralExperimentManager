"""
训练相关的数据类
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrainConfig:
    """训练配置"""
    params: Dict[str, Any]
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    feval_names: List[str] = field(default_factory=lambda: ["pearsonr_ic"])
    objective_name: str = "regression"
    seed: int = 42
    verbose_eval: int = 100
    
    def with_params(self, **kwargs) -> "TrainConfig":
        """返回新配置，覆盖部分参数"""
        new_params = {**self.params, **kwargs}
        return TrainConfig(
            params=new_params,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            feval_names=self.feval_names,
            objective_name=self.objective_name,
            seed=self.seed,
            verbose_eval=self.verbose_eval,
        )


@dataclass
class FitResult:
    """训练结果"""
    model: Any  # lgb.Booster or other
    evals_result: Dict[str, Dict[str, List[float]]]
    best_iteration: int
    params: Dict[str, Any]
    seed: int
    train_time: float = 0.0
    feature_importance: Optional[pd.DataFrame] = None


@dataclass 
class EvalResult:
    """评估结果"""
    metrics: Dict[str, float]
    series: Dict[str, pd.Series] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None


@dataclass
class MethodOutput:
    """
    Method 的完整输出
    
    Attributes:
        best_params: 最佳超参数 (来自 tuner)
        metrics_search: 搜索阶段指标
        metrics_eval: 评估阶段指标
        importance_vector: 特征重要性向量 (与当前 feature_names 对齐)
        feature_names_hash: 特征名哈希 (防错位)
        aux_state_delta: 辅助状态增量 (行业偏好等)
        model_artifacts: 模型产物路径
    """
    best_params: Dict[str, Any]
    metrics_eval: Dict[str, EvalResult]
    importance_vector: np.ndarray
    feature_names_hash: str
    metrics_search: Optional[Dict[str, float]] = None
    aux_state_delta: Optional[Dict[str, Any]] = None
    model_artifacts: Optional[Dict[str, Path]] = None
    fit_result: Optional[FitResult] = None
