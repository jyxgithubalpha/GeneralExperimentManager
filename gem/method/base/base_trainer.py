"""
BaseTrainer - 训练器基类

支持:
- 本地训练
- Ray Trainer 分布式训练
- 样本权重
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..method_dataclasses import FitResult, TrainConfig
from ...data.data_dataclasses import ProcessedViews


class BaseTrainer(ABC):
    """
    训练器基类
    
    子类需实现 fit() 方法，返回 FitResult
    """
    
    @abstractmethod
    def fit(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> "FitResult":
        """
        训练模型
        
        Args:
            views: 处理后的视图
            config: 训练配置
            mode: "full" 或 "tune"
            sample_weights: 可选的样本权重 {"train": ..., "val": ...}
            
        Returns:
            FitResult
        """
        pass
