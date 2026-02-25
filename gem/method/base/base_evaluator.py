"""
BaseEvaluator
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from ..method_dataclasses import EvalResult
from ...data.data_dataclasses import ProcessedViews


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        views: "ProcessedViews",
    ) -> Dict[str, "EvalResult"]:
        pass
