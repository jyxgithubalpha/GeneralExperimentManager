"""
BaseTuner - 调优器基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_trainer import BaseTrainer
    from ..training_dataclasses import TrainConfig
    from ...data.data_dataclasses import ProcessedViews


class BaseTuner(ABC):
    """调优器基类"""
    
    @abstractmethod
    def tune(
        self,
        views: "ProcessedViews",
        trainer: "BaseTrainer",
        config: "TrainConfig",
    ) -> Dict[str, Any]:
        """调参"""
        pass
