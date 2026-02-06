"""
StatePolicy - 控制 split 串/并行与更新频率

此模块现在是一个兼容层，从 policies 子模块重新导出所有类。
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

from .state_dataclasses import SplitResult, RollingState, StatePolicyMode

# 从 policies 子模块导入
from .policies import (
    StatePolicy,
    EMAStatePolicy,
    IndustryPreferencePolicy,
    StatePolicyFactory,
    StateUpdater,
)
from .policies.base import NoStatePolicy

# 从 bucketing 导入
from .bucketing import (
    quarter_bucket_fn,
    month_bucket_fn,
    BucketManager,
)

if TYPE_CHECKING:
    from .experiment_manager import StatePolicyConfig


# =============================================================================
# 便捷函数
# =============================================================================
# 3. EMAStatePolicy - EMA 更新策略
# =============================================================================

class EMAStatePolicy(StatePolicy):
    """
    EMA 状态策略
    
    使用指数移动平均更新特征重要性
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        topk: Optional[int] = None,
        normalize: bool = True,
    ):
        self.alpha = alpha
        self.topk = topk
        self.normalize = normalize
    
    def update_state(
        self,
        prev_state: RollingState,
        split_result: SplitResult,
    ) -> RollingState:
        """使用 EMA 更新状态"""
        new_state = RollingState(
            importance_ema=prev_state.importance_ema.copy() if prev_state.importance_ema is not None else None,
            feature_names_hash=prev_state.feature_names_hash,
            industry_preference=prev_state.industry_preference.copy() if prev_state.industry_preference else None,
            split_history=prev_state.split_history.copy(),
            metadata=prev_state.metadata.copy(),
        )
        
        # 更新 importance EMA
        new_importance = split_result.importance_vector
        new_state.update_importance(
            new_importance,
            alpha=self.alpha,
            feature_names_hash=split_result.feature_names_hash,
        )
        
        # 应用 topk 筛选
        if self.topk is not None and new_state.importance_ema is not None:
            mask = np.zeros_like(new_state.importance_ema)
            topk_indices = np.argsort(new_state.importance_ema)[-self.topk:]
            mask[topk_indices] = 1.0
            new_state.importance_ema = new_state.importance_ema * mask
        
        # 归一化
        if self.normalize and new_state.importance_ema is not None:
            total = np.sum(new_state.importance_ema)
            if total > 0:
                new_state.importance_ema = new_state.importance_ema / total
        
        # 更新行业偏好 (如果有)
        if split_result.industry_delta is not None:
            if new_state.industry_preference is None:
                new_state.industry_preference = split_result.industry_delta.copy()
            else:
                for k, v in split_result.industry_delta.items():
                    if k in new_state.industry_preference:
                        new_state.industry_preference[k] = (
                            self.alpha * v + (1 - self.alpha) * new_state.industry_preference[k]
                        )
                    else:
                        new_state.industry_preference[k] = v
        
        # 记录历史
        new_state.split_history.append(split_result.split_id)
        
        return new_state
    
    def aggregate_importance(
        self,
        results: List[SplitResult],
    ) -> np.ndarray:
        """聚合 bucket 内的重要性 (均值)"""
        if not results:
            return np.array([])
        
        # 验证 hash 一致
        hashes = set(r.feature_names_hash for r in results)
        if len(hashes) > 1:
            raise ValueError(f"Inconsistent feature names hashes in bucket: {hashes}")
        
        importance_list = [r.importance_vector for r in results]
        return np.mean(importance_list, axis=0)
    
    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        """从 bucket 聚合结果更新状态"""
        new_state = RollingState(
            importance_ema=prev_state.importance_ema.copy() if prev_state.importance_ema is not None else None,
            feature_names_hash=prev_state.feature_names_hash,
            industry_preference=prev_state.industry_preference.copy() if prev_state.industry_preference else None,
            split_history=prev_state.split_history.copy(),
            metadata=prev_state.metadata.copy(),
        )
        
        new_state.update_importance(
            agg_importance,
            alpha=self.alpha,
            feature_names_hash=feature_names_hash,
        )
        
        # 应用 topk 筛选
        if self.topk is not None and new_state.importance_ema is not None:
            mask = np.zeros_like(new_state.importance_ema)
            topk_indices = np.argsort(new_state.importance_ema)[-self.topk:]
            mask[topk_indices] = 1.0
            new_state.importance_ema = new_state.importance_ema * mask
        
        # 归一化
        if self.normalize and new_state.importance_ema is not None:
            total = np.sum(new_state.importance_ema)
            if total > 0:
                new_state.importance_ema = new_state.importance_ema / total
        
        return new_state


# =============================================================================
# 4. IndustryPreferencePolicy - 行业偏好策略
# =============================================================================

class IndustryPreferencePolicy(StatePolicy):
    """
    行业偏好策略
    
    跟踪行业收益，调整行业权重
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        industry_col: str = "industry",
    ):
        self.alpha = alpha
        self.industry_col = industry_col
        self._base_policy = EMAStatePolicy(alpha=alpha)
    
    def update_state(
        self,
        prev_state: RollingState,
        split_result: SplitResult,
    ) -> RollingState:
        """更新状态，包括行业偏好"""
        # 先用基础策略更新 importance
        new_state = self._base_policy.update_state(prev_state, split_result)
        
        # 更新行业偏好
        if split_result.industry_delta is not None:
            if new_state.industry_preference is None:
                new_state.industry_preference = {}
            
            for ind, delta in split_result.industry_delta.items():
                if ind in new_state.industry_preference:
                    new_state.industry_preference[ind] = (
                        self.alpha * delta + (1 - self.alpha) * new_state.industry_preference[ind]
                    )
                else:
                    new_state.industry_preference[ind] = delta
        
        return new_state
    
    def aggregate_importance(
        self,
        results: List[SplitResult],
    ) -> np.ndarray:
        return self._base_policy.aggregate_importance(results)
    
    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        return self._base_policy.update_state_from_bucket(
            prev_state, agg_importance, feature_names_hash
        )


# =============================================================================
# 5. StatePolicyFactory - 策略工厂
# =============================================================================

class StatePolicyFactory:
    """状态策略工厂"""
    
    @staticmethod
    def create(config: StatePolicyConfig) -> StatePolicy:
        """
        根据配置创建策略
        
        Args:
            config: 策略配置
            
        Returns:
            StatePolicy 实例
        """
        if config.mode == StatePolicyMode.NONE:
            return NoStatePolicy()
        elif config.mode in (StatePolicyMode.PER_SPLIT, StatePolicyMode.BUCKET):
            return EMAStatePolicy(
                alpha=config.ema_alpha,
                topk=config.importance_topk,
                normalize=config.normalize_importance,
            )
        else:
            return NoStatePolicy()


# =============================================================================
# 6. BucketManager - Bucket 管理
# =============================================================================

class BucketManager:
    """
    Bucket 管理器
    
    负责:
    - 将 splits 分组到 buckets
    - 管理 bucket 间的执行顺序
    """
    
    def __init__(
        self,
        bucket_fn: Optional[Callable[[SplitSpec], str]] = None,
    ):
        self.bucket_fn = bucket_fn or quarter_bucket_fn
    
    def group_splits(
        self,
        splitspecs: List[SplitSpec],
    ) -> Dict[str, List[SplitSpec]]:
        """
        将 splits 分组到 buckets
        
        Args:
            splitspecs: Split 规格列表
            
        Returns:
            {bucket_key: [SplitSpec, ...]} 字典
        """
        buckets: Dict[str, List[SplitSpec]] = {}
        
        for spec in splitspecs:
            key = self.bucket_fn(spec)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(spec)
        
        return buckets
    
    def get_bucket_order(
        self,
        buckets: Dict[str, List[SplitSpec]],
    ) -> List[str]:
        """
        获取 bucket 执行顺序 (按时间排序)
        
        Args:
            buckets: {bucket_key: [SplitSpec, ...]} 字典
            
        Returns:
            排序后的 bucket keys 列表
        """
        # 按 bucket 内最早的 test_start 排序
        def get_min_test_start(key: str) -> int:
            return min(s.test_start for s in buckets[key])
        
        return sorted(buckets.keys(), key=get_min_test_start)
    
    def create_execution_plan(
        self,
        splitspecs: List[SplitSpec],
        mode: StatePolicyMode,
    ) -> List[List[SplitSpec]]:
        """
        创建执行计划
        
        Args:
            splitspecs: Split 规格列表
            mode: 执行模式
            
        Returns:
            执行计划，每个元素是一个可并行执行的 split 组
        """
        if mode == StatePolicyMode.NONE.value:
            # 全并行
            return [splitspecs]
        elif mode == StatePolicyMode.PER_SPLIT.value:
            # 严格串行
            sorted_specs = sorted(splitspecs, key=lambda s: s.get_all_dates_range()[0])
            return [[s] for s in sorted_specs]
        elif mode == StatePolicyMode.BUCKET.value:
            # Bucket 内并行，Bucket 间串行
            buckets = self.group_splits(splitspecs)
            order = self.get_bucket_order(buckets)
            return [buckets[key] for key in order]
        else:
            return [splitspecs]


# =============================================================================
# 7. StateUpdater - 统一状态更新器
# =============================================================================

class StateUpdater:
    """
    统一状态更新器
    
    维护 RollingState 中的:
    - data_weight_state: 数据加权状态 (feature/industry EMA)
    - tuning_state: 调参状态 (last_best_params, history)
    
    支持 none/per_split/bucket 三种模式
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        topk: Optional[int] = None,
        normalize: bool = True,
        update_tuning_state: bool = True,
        shrink_ratio: float = 0.5,
    ):
        self.alpha = alpha
        self.topk = topk
        self.normalize = normalize
        self.update_tuning_state = update_tuning_state
        self.shrink_ratio = shrink_ratio
        self._ema_policy = EMAStatePolicy(alpha=alpha, topk=topk, normalize=normalize)
    
    def update(
        self,
        prev_state: RollingState,
        split_result: SplitResult,
        best_params: Optional[Dict[str, Any]] = None,
        best_objective: Optional[float] = None,
    ) -> RollingState:
        """
        更新状态
        
        Args:
            prev_state: 前一状态
            split_result: Split 结果
            best_params: 最佳参数 (用于更新 tuning_state)
            best_objective: 最佳目标值
            
        Returns:
            更新后的 RollingState
        """
        # 使用 EMA 策略更新 importance
        new_state = self._ema_policy.update_state(prev_state, split_result)
        
        # 更新 data_weight_state
        if new_state.data_weight_state is None:
            new_state.data_weight_state = DataWeightState()
        
        if new_state.importance_ema is not None:
            new_state.data_weight_state.feature_weights = new_state.importance_ema.copy()
            new_state.data_weight_state.feature_names_hash = new_state.feature_names_hash
        
        # 更新行业权重
        if split_result.industry_delta is not None:
            if new_state.data_weight_state.industry_weights is None:
                new_state.data_weight_state.industry_weights = {}
            for ind, delta in split_result.industry_delta.items():
                if ind in new_state.data_weight_state.industry_weights:
                    new_state.data_weight_state.industry_weights[ind] = (
                        self.alpha * delta + 
                        (1 - self.alpha) * new_state.data_weight_state.industry_weights[ind]
                    )
                else:
                    new_state.data_weight_state.industry_weights[ind] = delta
        
        # 更新 tuning_state
        if self.update_tuning_state and best_params is not None:
            if new_state.tuning_state is None:
                new_state.tuning_state = TuningState(search_space_shrink_ratio=self.shrink_ratio)
            new_state.tuning_state.update(best_params, best_objective or 0.0)
        
        return new_state
    
    def aggregate(
        self,
        split_results: List[SplitResult],
        best_params_list: Optional[List[Dict[str, Any]]] = None,
        best_objectives: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        聚合 bucket 内的结果
        
        Args:
            split_results: Split 结果列表
            best_params_list: 最佳参数列表
            best_objectives: 最佳目标值列表
            
        Returns:
            聚合结果字典
        """
        # 聚合 importance
        agg_importance = self._ema_policy.aggregate_importance(split_results)
        
        # 聚合 industry_delta
        agg_industry = {}
        for result in split_results:
            if result.industry_delta:
                for ind, delta in result.industry_delta.items():
                    if ind not in agg_industry:
                        agg_industry[ind] = []
                    agg_industry[ind].append(delta)
        
        agg_industry_mean = {k: np.mean(v) for k, v in agg_industry.items()}
        
        # 聚合 best_params (选择最好的)
        agg_best_params = None
        agg_best_objective = None
        if best_params_list and best_objectives:
            best_idx = np.argmax(best_objectives)
            agg_best_params = best_params_list[best_idx]
            agg_best_objective = best_objectives[best_idx]
        
        return {
            "importance": agg_importance,
            "industry_delta": agg_industry_mean,
            "best_params": agg_best_params,
            "best_objective": agg_best_objective,
            "feature_names_hash": split_results[0].feature_names_hash if split_results else "",
        }
    
    def update_from_aggregate(
        self,
        prev_state: RollingState,
        agg_result: Dict[str, Any],
    ) -> RollingState:
        """
        从聚合结果更新状态
        
        Args:
            prev_state: 前一状态
            agg_result: 聚合结果
            
        Returns:
            更新后的 RollingState
        """
        new_state = self._ema_policy.update_state_from_bucket(
            prev_state,
            agg_result["importance"],
            agg_result["feature_names_hash"],
        )
        
        # 更新 data_weight_state
        if new_state.data_weight_state is None:
            new_state.data_weight_state = DataWeightState()
        
        if new_state.importance_ema is not None:
            new_state.data_weight_state.feature_weights = new_state.importance_ema.copy()
            new_state.data_weight_state.feature_names_hash = new_state.feature_names_hash
        
        # 更新行业权重
        if agg_result.get("industry_delta"):
            if new_state.data_weight_state.industry_weights is None:
                new_state.data_weight_state.industry_weights = {}
            for ind, delta in agg_result["industry_delta"].items():
                if ind in new_state.data_weight_state.industry_weights:
                    new_state.data_weight_state.industry_weights[ind] = (
                        self.alpha * delta + 
                        (1 - self.alpha) * new_state.data_weight_state.industry_weights[ind]
                    )
                else:
                    new_state.data_weight_state.industry_weights[ind] = delta
        
        # 更新 tuning_state
        if self.update_tuning_state and agg_result.get("best_params"):
            if new_state.tuning_state is None:
                new_state.tuning_state = TuningState(search_space_shrink_ratio=self.shrink_ratio)
            new_state.tuning_state.update(
                agg_result["best_params"],
                agg_result.get("best_objective", 0.0),
            )
        
        return new_state


# =============================================================================
# 8. 便捷函数
# =============================================================================

def update_state(
    prev_state: Optional[RollingState],
    split_result: SplitResult,
    config: StatePolicyConfig,
) -> RollingState:
    """
    更新状态 (便捷函数)
    
    Args:
        prev_state: 前一状态 (可为 None)
        split_result: Split 结果
        config: 策略配置
        
    Returns:
        新状态
    """
    if prev_state is None:
        prev_state = RollingState()
    
    policy = StatePolicyFactory.create(config)
    return policy.update_state(prev_state, split_result)


def aggregate_bucket_results(
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> np.ndarray:
    """
    聚合 bucket 结果 (便捷函数)
    """
    policy = StatePolicyFactory.create(config)
    return policy.aggregate_importance(results)


def update_state_from_bucket_results(
    prev_state: Optional[RollingState],
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> RollingState:
    """
    从 bucket 结果更新状态 (便捷函数)
    """
    if prev_state is None:
        prev_state = RollingState()
    
    policy = StatePolicyFactory.create(config)
    agg_importance = policy.aggregate_importance(results)
    
    if len(results) > 0:
        feature_names_hash = results[0].feature_names_hash
    else:
        feature_names_hash = ""
    
    return policy.update_state_from_bucket(prev_state, agg_importance, feature_names_hash)
