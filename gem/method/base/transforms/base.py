"""
BaseTransform - Transform base class.
"""


from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl

from ...method_dataclasses import TransformState

TransformContext = Dict[str, Any]


def extract_date_keys(
    keys: Optional["pl.DataFrame"],
    date_col: str = "date",
) -> Optional[np.ndarray]:
    """Extract date column from keys DataFrame."""
    if keys is None or date_col not in keys.columns:
        return None
    return keys[date_col].to_numpy()


class BaseTransform(ABC):
    """
    Transform base class.
    
    Supports receiving external parameters through context (e.g., feature weights from RollingState).
    Subclasses can access these parameters via `self._context`.
    """
    
    def __init__(self):
        self._state: Optional[TransformState] = None
        self._context: TransformContext = {}
    
    def set_context(self, context: TransformContext) -> "BaseTransform":
        """Set context parameters (called by Pipeline)."""
        self._context = context
        return self
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self._context.get(key, default)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "BaseTransform":
        """
        Fit transform.
        
        Args:
            X: Feature array
            y: Labels
            keys: Optional key array (e.g., date) for grouped computation
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transform.
        
        Returns:
            (X_transformed, y_transformed)
        """
        pass
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform."""
        self.fit(X, y, keys)
        return self.transform(X, y, keys)
    
    def inverse_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse transform (optional implementation)."""
        return X, y
    
    @property
    def state(self) -> Optional[TransformState]:
        return self._state
