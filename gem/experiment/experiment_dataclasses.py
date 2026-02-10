"""
状态管理相关的数据结构定义

架构设计:
- BaseState: 状态基类，所有可插拔状态继承此类
- RollingState: 状态容器，管理多个 BaseState 实例
- 具体状态类: FeatureImportanceState, TuningState, DataWeightState 等
"""

import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
from enum import Enum
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from ..data.data_dataclasses import SplitSpec
from ..method.method_dataclasses import TrainConfig


# =============================================================================
# 状态基类
# =============================================================================


class BaseState(ABC):
    """
    可插拔状态基类
    
    所有可在 RollingState 中存储的状态组件都应继承此类。
    每个 State 类应定义自己的 state_key 作为唯一标识。
    """
    
    @classmethod
    @abstractmethod
    def state_key(cls) -> str:
        """状态的唯一标识符"""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """更新状态"""
        pass
    
    def to_transform_context(self) -> Dict[str, Any]:
        """
        转换为 transform 可用的上下文
        
        返回的 dict 会被传递给 Transform.fit/transform 方法
        """
        return {}


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
    mode: str = "none"
    bucket_fn: Optional[Callable[[SplitSpec], str]] = None
    ema_alpha: float = 0.3
    importance_topk: Optional[int] = None
    normalize_importance: bool = True


@dataclass
class ResourceRequest:
    """资源请求"""
    trial_gpus: float = 1.0  # 每个 trial 的 GPU 数量
    final_train_gpus: float = 1.0  # 最终训练的 GPU 数量
    trial_cpus: float = 1.0
    final_train_cpus: float = 1.0


@dataclass
class ExperimentConfig:
    """
    实验配置
    """
    name: str
    output_dir: Path
    state_policy: "StatePolicyConfig" = None  # type: ignore  # Set via Hydra
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
    Split 任务 - ExperimentManager 构建
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
    train_config: Optional["TrainConfig"] = None


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


# =============================================================================
# 具体状态类
# =============================================================================

@dataclass
class FeatureImportanceState(BaseState):
    """
    特征重要性状态
    
    用于存储和更新特征重要性的 EMA，可传递给 FeatureWeightTransform。
    """
    importance_ema: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    feature_names_hash: Optional[str] = None
    alpha: float = 0.3
    
    @classmethod
    def state_key(cls) -> str:
        return "feature_importance"
    
    def update(
        self,
        new_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """更新特征重要性 EMA"""
        if new_importance is None:
            return
        
        # 更新 alpha
        if alpha is not None:
            self.alpha = alpha
        
        # 检查特征名一致性
        if feature_names is not None:
            new_hash = hashlib.md5(str(feature_names).encode()).hexdigest()[:8]
            if self.feature_names_hash is not None and self.feature_names_hash != new_hash:
                raise ValueError(f"Feature names hash mismatch: {self.feature_names_hash} vs {new_hash}")
            self.feature_names = feature_names
            self.feature_names_hash = new_hash
        
        # 更新 EMA
        if self.importance_ema is None:
            self.importance_ema = new_importance.copy()
        else:
            self.importance_ema = self.alpha * new_importance + (1 - self.alpha) * self.importance_ema
    
    def to_transform_context(self) -> Dict[str, Any]:
        """feature_weights 传递给 FeatureWeightTransform"""
        return {
            "feature_weights": self.importance_ema,
            "feature_names": self.feature_names,
        }
    
    def get_topk_indices(self, k: int) -> np.ndarray:
        """获取 top-k 重要特征的索引"""
        if self.importance_ema is None:
            return np.array([])
        k = min(k, len(self.importance_ema))
        return np.argsort(self.importance_ema)[-k:]


@dataclass
class SampleWeightState(BaseState):
    """
    样本加权状态
    
    用于计算训练样本的权重，支持按资产/时间/行业加权。
    """
    asset_weights: Optional[Dict[str, float]] = None
    industry_weights: Optional[Dict[str, float]] = None
    time_weights: Optional[Dict[int, float]] = None
    
    @classmethod
    def state_key(cls) -> str:
        return "sample_weight"
    
    def update(
        self,
        asset_weights: Optional[Dict[str, float]] = None,
        industry_weights: Optional[Dict[str, float]] = None,
        time_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """更新样本权重"""
        if asset_weights is not None:
            self.asset_weights = asset_weights
        if industry_weights is not None:
            self.industry_weights = industry_weights
        if time_weights is not None:
            self.time_weights = time_weights
    
    def to_transform_context(self) -> Dict[str, Any]:
        return {
            "asset_weights": self.asset_weights,
            "industry_weights": self.industry_weights,
            "time_weights": self.time_weights,
        }
    
    def get_sample_weight(
        self,
        keys: pl.DataFrame,
        group: Optional[pl.DataFrame] = None,
        industry_col: str = "industry",
    ) -> np.ndarray:
        """计算样本权重"""
        n = keys.height
        weights = np.ones(n, dtype=np.float32)
        
        if self.asset_weights is not None:
            codes = keys["code"].to_numpy()
            for i, code in enumerate(codes):
                if code in self.asset_weights:
                    weights[i] *= self.asset_weights[code]
        
        if self.time_weights is not None:
            dates = keys["date"].to_numpy()
            for i, d in enumerate(dates):
                if d in self.time_weights:
                    weights[i] *= self.time_weights[d]
        
        if self.industry_weights is not None and group is not None:
            if industry_col in group.columns:
                industries = group[industry_col].to_numpy()
                for i, ind in enumerate(industries):
                    if ind in self.industry_weights:
                        weights[i] *= self.industry_weights[ind]
        
        return weights


@dataclass
class TuningState(BaseState):
    """
    调参状态 - 用于超参搜索优化
    """
    last_best_params: Optional[Dict[str, Any]] = None
    params_history: List[Dict[str, Any]] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    search_space_shrink_ratio: float = 0.5
    
    @classmethod
    def state_key(cls) -> str:
        return "tuning"
    
    def update(self, best_params: Dict[str, Any], best_objective: float) -> None:
        """更新调参状态"""
        self.last_best_params = best_params.copy()
        self.params_history.append(best_params.copy())
        self.objective_history.append(best_objective)
    
    def to_transform_context(self) -> Dict[str, Any]:
        return {
            "last_best_params": self.last_best_params,
        }
    
    def get_shrunk_space(
        self,
        base_space: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """根据历史收缩搜索空间"""
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


# =============================================================================
# 状态容器
# =============================================================================

@dataclass
class RollingState:
    """
    滚动状态容器 - 管理多个可插拔的 State 组件
    
    支持动态注册和获取不同类型的 State，并能将所有 State 
    转换为统一的 transform context 传递给 Transform Pipeline。
    
    Attributes:
        states: 状态字典 {state_key: BaseState}
        split_history: 历史 split ID 列表
        metadata: 额外元数据
    """
    states: Dict[str, BaseState] = field(default_factory=dict)
    split_history: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def register_state(self, state: BaseState) -> None:
        """注册一个状态组件"""
        self.states[state.state_key()] = state
    
    def get_state(self, state_class: Type[BaseState]) -> Optional[BaseState]:
        """获取指定类型的状态"""
        key = state_class.state_key()
        return self.states.get(key)
    
    def get_or_create_state(self, state_class: Type[BaseState]) -> BaseState:
        """获取或创建指定类型的状态"""
        key = state_class.state_key()
        if key not in self.states:
            self.states[key] = state_class()
        return self.states[key]  # type: ignore
    
    def has_state(self, state_class: Type[BaseState]) -> bool:
        """检查是否存在指定类型的状态"""
        return state_class.state_key() in self.states
    
    def to_transform_context(self) -> Dict[str, Any]:
        """
        将所有状态转换为 transform context
        
        返回的 dict 可以传递给 BaseTransformPipeline.process_views()
        """
        context = {}
        for state in self.states.values():
            context.update(state.to_transform_context())
        return context
    
    # =========================================================================
    # 便捷方法 - 常用操作的快捷方式
    # =========================================================================
    
    def update_importance(
        self,
        new_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.3,
    ) -> None:
        """更新特征重要性 (便捷方法)"""
        state = self.get_or_create_state(FeatureImportanceState)
        state.update(new_importance, feature_names, alpha)
    
    def update_tuning(
        self,
        best_params: Dict[str, Any],
        best_objective: float,
    ) -> None:
        """更新调参状态 (便捷方法)"""
        state = self.get_or_create_state(TuningState)
        state.update(best_params, best_objective)
    
    @property
    def feature_importance(self) -> Optional[FeatureImportanceState]:
        """获取特征重要性状态 (便捷属性)"""
        return self.get_state(FeatureImportanceState)
    
    @property
    def tuning(self) -> Optional[TuningState]:
        """获取调参状态 (便捷属性)"""
        return self.get_state(TuningState)
    
    @property
    def sample_weight(self) -> Optional[SampleWeightState]:
        """获取样本权重状态 (便捷属性)"""
        return self.get_state(SampleWeightState)
    
    # =========================================================================
    # 向后兼容属性 (deprecated, 建议使用新 API)
    # =========================================================================
    
    @property
    def importance_ema(self) -> Optional[np.ndarray]:
        """[deprecated] 使用 feature_importance.importance_ema 代替"""
        state = self.get_state(FeatureImportanceState)
        return state.importance_ema if state else None
    
    @property
    def tuning_state(self) -> Optional[TuningState]:
        """[deprecated] 使用 tuning 属性代替"""
        return self.get_state(TuningState)
    
    @property
    def data_weight_state(self) -> Optional[SampleWeightState]:
        """[deprecated] 使用 sample_weight 属性代替"""
        return self.get_state(SampleWeightState)
    
    # =========================================================================
    # 序列化
    # =========================================================================
    
    def save(self, path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path) -> "RollingState":
        with open(path, 'rb') as f:
            return pickle.load(f)
