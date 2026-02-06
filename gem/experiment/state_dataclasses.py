"""
状态管理相关的数据类
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..data.data_dataclasses import SplitSpec

class StatePolicyMode(Enum):
    """状态策略模式"""
    NONE = "none"           # 全并行，无状态传递
    PER_SPLIT = "per_split" # 严格串行，每个 split 后更新状态
    BUCKET = "bucket"       # Bucket 内并行，Bucket 间串行


@dataclass
class StatePolicyConfig:
    """
    状态策略配置
    
    Attributes:
        mode: 策略模式
        bucket_fn: Bucket 分组函数
        ema_alpha: EMA 平滑系数
        importance_topk: 只保留 top-k 特征
        normalize_importance: 是否归一化重要性
    """
    mode: StatePolicyMode = StatePolicyMode.NONE
    bucket_fn: Optional[Callable[[SplitSpec], str]] = None
    ema_alpha: float = 0.3
    importance_topk: Optional[int] = None
    normalize_importance: bool = True


@dataclass
class ResourceRequest:
    """资源请求"""
    trial_gpus: float = 1.0  # 每个 trial 的 GPU 数
    final_train_gpus: float = 1.0  # 最终训练的 GPU 数
    trial_cpus: float = 1.0
    final_train_cpus: float = 1.0


@dataclass
class ExperimentConfig:
    """
    实验配置
    """
    name: str
    output_dir: Path
    state_policy: StatePolicyConfig = field(default_factory=StatePolicyConfig)
    n_trials: int = 50
    trial_timeout: Optional[int] = None
    parallel_trials: int = 1
    seed: int = 42
    ray_address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    use_ray: bool = False
    resource_request: Optional[ResourceRequest] = None


@dataclass
class SplitTask:
    """
    Split 任务 - ExperimentManager 构造
    
    Attributes:
        split_id: Split ID
        splitspec: Split 规格
        seed: 随机种子
        resource_request: 资源请求
    """
    split_id: int
    splitspec: SplitSpec
    seed: int = 42
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    train_config: Optional[TrainConfig] = None


@dataclass
class SplitResult:
    """
    Split 结果 - 单次 split 训练的结果
    
    Attributes:
        split_id: Split 标识
        importance_vector: 特征重要性向量
        feature_names_hash: 特征名哈希
        industry_delta: 行业增量权重
        metrics: 评估指标
        best_params: 最佳超参数
        best_objective: 最佳目标值
        skipped: 是否被跳过
        skip_reason: 跳过原因
    """
    split_id: int
    importance_vector: Optional[np.ndarray] = None
    feature_names_hash: Optional[str] = None
    industry_delta: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_objective: Optional[float] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class DataWeightState:
    """
    数据加权状态 - 用于 DataProcessor
    
    Attributes:
        feature_weights: 特征权重 (从 importance EMA 更新)
        asset_weights: 股票权重 {code: weight}
        industry_weights: 行业权重 {industry: weight}
        time_weights: 时间权重 {date: weight}
        feature_names_hash: 特征名哈希 (防错位)
    """
    feature_weights: Optional[np.ndarray] = None
    asset_weights: Optional[Dict[str, float]] = None
    industry_weights: Optional[Dict[str, float]] = None
    time_weights: Optional[Dict[int, float]] = None
    feature_names_hash: Optional[str] = None
    
    def get_sample_weight(
        self,
        keys: pd.DataFrame,
        group: Optional[pd.DataFrame] = None,
        industry_col: str = "industry",
    ) -> np.ndarray:
        """
        计算样本权重
        
        Args:
            keys: 主键 DataFrame，包含 date, code
            group: 分组 DataFrame，包含 industry 等
            industry_col: 行业列名
            
        Returns:
            样本权重数组 [n_samples]
        """
        n = len(keys)
        weights = np.ones(n, dtype=np.float32)
        
        # 应用资产权重
        if self.asset_weights is not None:
            codes = keys["code"].values
            for i, code in enumerate(codes):
                if code in self.asset_weights:
                    weights[i] *= self.asset_weights[code]
        
        # 应用时间权重
        if self.time_weights is not None:
            dates = keys["date"].values
            for i, d in enumerate(dates):
                if d in self.time_weights:
                    weights[i] *= self.time_weights[d]
        
        # 应用行业权重
        if self.industry_weights is not None and group is not None:
            if industry_col in group.columns:
                industries = group[industry_col].values
                for i, ind in enumerate(industries):
                    if ind in self.industry_weights:
                        weights[i] *= self.industry_weights[ind]
        
        return weights


@dataclass
class TuningState:
    """
    调参状态 - 用于超参搜索优化
    
    Attributes:
        last_best_params: 上一 split 的最佳参数
        params_history: 历史最佳参数列表
        objective_history: 历史最佳目标值
        search_space_shrink_ratio: 搜索空间收缩比例
    """
    last_best_params: Optional[Dict[str, Any]] = None
    params_history: List[Dict[str, Any]] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    search_space_shrink_ratio: float = 0.5  # 收缩到原来的50%
    
    def update(self, best_params: Dict[str, Any], best_objective: float) -> None:
        """更新调参状态"""
        self.last_best_params = best_params.copy()
        self.params_history.append(best_params.copy())
        self.objective_history.append(best_objective)
    
    def get_shrunk_space(
        self,
        base_space: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """
        根据历史收缩搜索空间
        
        Args:
            base_space: 基础搜索空间 {param_name: (low, high)}
            
        Returns:
            收缩后的搜索空间
        """
        if self.last_best_params is None:
            return base_space
        
        shrunk_space = {}
        ratio = self.search_space_shrink_ratio
        
        for name, (low, high) in base_space.items():
            if name in self.last_best_params:
                center = self.last_best_params[name]
                half_range = (high - low) * ratio / 2
                new_low = max(low, center - half_range)
                new_high = min(high, center + half_range)
                shrunk_space[name] = (new_low, new_high)
            else:
                shrunk_space[name] = (low, high)
        
        return shrunk_space


@dataclass
class RollingState:
    """
    滚动状态 - split 间传递的状态
    
    Attributes:
        importance_ema: 特征重要性的 EMA
        feature_names_hash: 特征名哈希 (防错位)
        industry_preference: 行业偏好权重
        split_history: 历史 split 的摘要
        data_weight_state: 数据加权状态 (用于 DataProcessor)
        tuning_state: 调参状态 (用于 Tuner)
        metadata: 额外元数据
    """
    importance_ema: Optional[np.ndarray] = None
    feature_names_hash: Optional[str] = None
    industry_preference: Optional[Dict[str, float]] = None
    split_history: List[int] = field(default_factory=list)
    data_weight_state: Optional[DataWeightState] = None
    tuning_state: Optional[TuningState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_importance(self, new_importance: np.ndarray, alpha: float = 0.3,
                         feature_names_hash: Optional[str] = None) -> None:
        """更新重要性 EMA"""
        if feature_names_hash is not None:
            if self.feature_names_hash is not None and self.feature_names_hash != feature_names_hash:
                raise ValueError(f"Feature names hash mismatch: {self.feature_names_hash} vs {feature_names_hash}")
            self.feature_names_hash = feature_names_hash
        
        if self.importance_ema is None:
            self.importance_ema = new_importance.copy()
        else:
            self.importance_ema = alpha * new_importance + (1 - alpha) * self.importance_ema
        
        # 同步更新 data_weight_state
        if self.data_weight_state is None:
            self.data_weight_state = DataWeightState()
        self.data_weight_state.feature_weights = self.importance_ema.copy()
        self.data_weight_state.feature_names_hash = self.feature_names_hash
    
    def update_tuning_state(self, best_params: Dict[str, Any], best_objective: float) -> None:
        """更新调参状态"""
        if self.tuning_state is None:
            self.tuning_state = TuningState()
        self.tuning_state.update(best_params, best_objective)
    
    def save(self, path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path) -> "RollingState":
        with open(path, 'rb') as f:
            return pickle.load(f)
