"""
StatePolicy - 控制 split 的并行与更新频率

此模块现在是一个兼容层，从 policies 子模块重新导出所有类。
"""



from typing import Any, Dict, List, Optional

import numpy as np

from .experiment_dataclasses import (
    SplitResult,
    RollingState,
    SampleWeightState,
    TuningState,
    StatePolicyConfig,
    FeatureImportanceState,
)

# Backward compatibility
DataWeightState = SampleWeightState
from abc import ABC, abstractmethod


class StatePolicy(ABC):
    """状态策略基类"""
    
    @abstractmethod
    def update_state(
        self,
        prev_state: RollingState,
        split_result: SplitResult,
    ) -> RollingState:
        """更新状态"""
        pass
    
    @abstractmethod
    def aggregate_importance(
        self,
        results: List[SplitResult],
    ) -> np.ndarray:
        """聚合重要性"""
        pass
    
    @abstractmethod
    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        """从 bucket 结果更新状态"""
        pass


class NoStatePolicy(StatePolicy):
    """无状态策略 - 不进行任何状态更新"""
    
    def update_state(
        self,
        prev_state: RollingState,
        split_result: SplitResult,
    ) -> RollingState:
        return prev_state
    
    def aggregate_importance(
        self,
        results: List[SplitResult],
    ) -> np.ndarray:
        if not results:
            return np.array([])
        return np.mean([r.importance_vector for r in results], axis=0)
    
    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        return prev_state


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
        import copy
        new_state = RollingState(
            states=copy.deepcopy(prev_state.states),
            split_history=prev_state.split_history.copy(),
            metadata=prev_state.metadata.copy(),
        )
        
        # 更新 importance EMA
        new_importance = split_result.importance_vector
        if new_importance is not None:
            new_state.update_importance(
                new_importance,
                alpha=self.alpha,
            )
            
            # 应用 topk 筛选和归一化
            fi_state = new_state.get_state(FeatureImportanceState)
            if fi_state is not None and fi_state.importance_ema is not None:
                if self.topk is not None:
                    mask = np.zeros_like(fi_state.importance_ema)
                    topk_indices = np.argsort(fi_state.importance_ema)[-self.topk:]
                    mask[topk_indices] = 1.0
                    fi_state.importance_ema = fi_state.importance_ema * mask
                
                if self.normalize:
                    total = np.sum(fi_state.importance_ema)
                    if total > 0:
                        fi_state.importance_ema = fi_state.importance_ema / total
        
        # 更新行业偏好 (如果有)
        if split_result.industry_delta is not None:
            sw_state = new_state.get_or_create_state(SampleWeightState)
            if sw_state.industry_weights is None:
                sw_state.industry_weights = split_result.industry_delta.copy()
            else:
                for k, v in split_result.industry_delta.items():
                    if k in sw_state.industry_weights:
                        sw_state.industry_weights[k] = (
                            self.alpha * v + (1 - self.alpha) * sw_state.industry_weights[k]
                        )
                    else:
                        sw_state.industry_weights[k] = v
        
        # 记录历史
        new_state.split_history.append(split_result.split_id)
        
        return new_state
    
    def aggregate_importance(
        self,
        results: List[SplitResult],
    ) -> np.ndarray:
        """聚合 bucket 内的重要性(均值)"""
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
        import copy
        new_state = RollingState(
            states=copy.deepcopy(prev_state.states),
            split_history=prev_state.split_history.copy(),
            metadata=prev_state.metadata.copy(),
        )
        
        if agg_importance is not None and len(agg_importance) > 0:
            new_state.update_importance(
                agg_importance,
                alpha=self.alpha,
            )
            
            # 应用 topk 筛选和归一化
            fi_state = new_state.get_state(FeatureImportanceState)
            if fi_state is not None and fi_state.importance_ema is not None:
                if self.topk is not None:
                    mask = np.zeros_like(fi_state.importance_ema)
                    topk_indices = np.argsort(fi_state.importance_ema)[-self.topk:]
                    mask[topk_indices] = 1.0
                    fi_state.importance_ema = fi_state.importance_ema * mask
                
                if self.normalize:
                    total = np.sum(fi_state.importance_ema)
                    if total > 0:
                        fi_state.importance_ema = fi_state.importance_ema / total
        
        return new_state


class StatePolicyFactory:
    """状态策略工厂"""
    
    @staticmethod
    def create(config: StatePolicyConfig) -> StatePolicy:
        if config.mode == "none":
            return NoStatePolicy()
        elif config.mode in ("per_split", "bucket"):
            return EMAStatePolicy(
                alpha=config.ema_alpha,
                topk=config.importance_topk,
                normalize=config.normalize_importance,
            )
        else:
            return NoStatePolicy()


class StateUpdater:
    """
    统一状态更新器
    
    维护 RollingState 中的:
    - data_weight_state: 数据加权状态(feature/industry EMA)
    - tuning_state: 调参状态(last_best_params, history)
    
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
            best_params: 最佳参数(用于更新 tuning_state)
            best_objective: 最佳目标值
            
        Returns:
            更新后的 RollingState
        """
        # 使用 EMA 策略更新 importance
        new_state = self._ema_policy.update_state(prev_state, split_result)
        
        # 更新行业权重
        if split_result.industry_delta is not None:
            sw_state = new_state.get_or_create_state(SampleWeightState)
            if sw_state.industry_weights is None:
                sw_state.industry_weights = {}
            for ind, delta in split_result.industry_delta.items():
                if ind in sw_state.industry_weights:
                    sw_state.industry_weights[ind] = (
                        self.alpha * delta + 
                        (1 - self.alpha) * sw_state.industry_weights[ind]
                    )
                else:
                    sw_state.industry_weights[ind] = delta
        
        # 更新 tuning_state
        if self.update_tuning_state and best_params is not None:
            new_state.update_tuning(best_params, best_objective or 0.0)
        
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
        
        # 更新行业权重
        if agg_result.get("industry_delta"):
            sw_state = new_state.get_or_create_state(SampleWeightState)
            if sw_state.industry_weights is None:
                sw_state.industry_weights = {}
            for ind, delta in agg_result["industry_delta"].items():
                if ind in sw_state.industry_weights:
                    sw_state.industry_weights[ind] = (
                        self.alpha * delta + 
                        (1 - self.alpha) * sw_state.industry_weights[ind]
                    )
                else:
                    sw_state.industry_weights[ind] = delta
        
        # 更新 tuning_state
        if self.update_tuning_state and agg_result.get("best_params"):
            new_state.update_tuning(
                agg_result["best_params"],
                agg_result.get("best_objective", 0.0),
            )
        
        return new_state


def update_state(
    prev_state: Optional[RollingState],
    split_result: SplitResult,
    config: StatePolicyConfig,
) -> RollingState:
    """
    更新状态(便捷函数)
    
    Args:
        prev_state: 前一状态(可为 None)
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
    从 bucket 结果更新状态(便捷函数)
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
