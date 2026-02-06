"""
StateUpdater - 统一状态更新器
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..state_dataclasses import DataWeightState, RollingState, TuningState, SplitResult
from .ema import EMAStatePolicy


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
