"""
IndustryPreferencePolicy - 行业偏好策略
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..state_dataclasses import RollingState, SplitResult
from .base import StatePolicy
from .ema import EMAStatePolicy


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
