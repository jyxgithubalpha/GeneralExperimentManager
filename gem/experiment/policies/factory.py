"""
StatePolicyFactory - 策略工厂
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import StatePolicy, NoStatePolicy
from .ema import EMAStatePolicy

if TYPE_CHECKING:
    from ..state_dataclasses import StatePolicyConfig, StatePolicyMode


class StatePolicyFactory:
    """状态策略工厂"""
    
    @staticmethod
    def create(config: "StatePolicyConfig") -> StatePolicy:
        """
        根据配置创建策略
        
        Args:
            config: 策略配置
            
        Returns:
            StatePolicy 实例
        """
        from ..state_dataclasses import StatePolicyMode
        
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
