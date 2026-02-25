"""
BaseTrainer - Trainer base class.

Supports:
- Local training
- Ray Trainer distributed training
- Sample weights
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..method_dataclasses import FitResult, TrainConfig
from ...data.data_dataclasses import ProcessedViews


class BaseTrainer(ABC):
    """
    Trainer base class.

    Subclasses must implement fit() method, returning FitResult.
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
        Train model.

        Args:
            views: Processed data views
            config: Training configuration
            mode: Training mode ("full" or "tune")
            sample_weights: Optional sample weights dict

        Returns:
            FitResult with trained model
        """
        pass
