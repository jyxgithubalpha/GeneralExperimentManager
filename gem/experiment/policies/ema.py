"""
EMAStatePolicy - EMA 状态更新策略
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..state_dataclasses import RollingState, SplitResult
from .base import StatePolicy


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
