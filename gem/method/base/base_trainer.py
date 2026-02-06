"""
BaseTrainer - 训练器基类
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..training_dataclasses import FitResult, TrainConfig
    from ...data.data_dataclasses import ProcessedViews


class BaseTrainer(ABC):
    """训练器基类"""
    
    @abstractmethod
    def fit(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        mode: str = "full",
    ) -> "FitResult":
        """训练模型"""
        pass
