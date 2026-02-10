"""
BaseTuner - 调优器基类

支持:
- Optuna 串行/并行搜索
- Ray Tune 并行搜索
- RollingState 热启动和搜索空间收缩
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ...experiment.experiment_dataclasses import RollingState

from .base_trainer import BaseTrainer
from ..method_dataclasses import TrainConfig, TuneResult
from ...data.data_dataclasses import ProcessedViews


class BaseTuner(ABC):
    """
    调优器基类
    
    子类需实现 tune() 方法，返回 TuneResult
    """
    
    @abstractmethod
    def tune(
        self,
        views: "ProcessedViews",
        trainer: "BaseTrainer",
        config: "TrainConfig",
        rolling_state: Optional["RollingState"] = None,
        **kwargs,
    ) -> TuneResult:
        """
        执行超参搜索
        
        Args:
            views: 处理后的视图
            trainer: 训练器
            config: 训练配置
            rolling_state: 滚动状态 (用于热启动和搜索空间收缩)
            
        Returns:
            TuneResult
        """
        pass
