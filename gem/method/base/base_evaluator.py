"""
BaseEvaluator - 评估器基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..training_dataclasses import EvalResult
    from ...data.data_dataclasses import ProcessedViews


class BaseEvaluator(ABC):
    """评估器基类"""
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        views: "ProcessedViews",
    ) -> Dict[str, "EvalResult"]:
        """评估模型"""
        pass
