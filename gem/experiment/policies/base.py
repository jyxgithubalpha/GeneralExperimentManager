"""
StatePolicy Base - 状态策略基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..state_dataclasses import RollingState, SplitResult


class StatePolicy(ABC):
    """
    状态策略基类
    
    控制:
    - split 间的状态传递
    - 执行模式 (全并行/串行/bucket)
    """
    
    @abstractmethod
    def update_state(
        self,
        prev_state: RollingState,
        split_result: SplitResult,
    ) -> RollingState:
        """更新状态 (per_split 模式)"""
        pass
    
    @abstractmethod
    def aggregate_importance(
        self,
        results: List[SplitResult],
    ) -> np.ndarray:
        """聚合 bucket 内的重要性"""
        pass
    
    @abstractmethod
    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        """从 bucket 聚合结果更新状态"""
        pass


class NoStatePolicy(StatePolicy):
    """
    无状态策略 - 全并行，无状态传递
    """
    
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
        return np.array([])
    
    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        return prev_state
