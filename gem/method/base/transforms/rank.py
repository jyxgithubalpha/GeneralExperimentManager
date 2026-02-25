"""
RankTransform - Rank transformation.
"""


from typing import Literal, Optional, Tuple

import numpy as np

from ...method_dataclasses import TransformState
from .base import BaseTransform


class RankTransform(BaseTransform):
    """Rank transformation."""
    
    def __init__(
        self,
        target: Literal["X", "y", "both"] = "X",
        per_date: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.target = target
        self.per_date = per_date
        self.normalize = normalize
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "RankTransform":
        self._state = TransformState(stats={"per_date": self.per_date, "normalize": self.normalize})
        return self
    
    def _rank_array(self, arr: np.ndarray, keys: Optional[np.ndarray]) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float32)
        
        def rank_1d(x: np.ndarray) -> np.ndarray:
            ranks = np.argsort(np.argsort(x)).astype(np.float32)
            if self.normalize:
                ranks = ranks / (len(x) - 1) if len(x) > 1 else ranks
            return ranks
        
        if self.per_date and keys is not None:
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = arr[mask]
                if sub.ndim > 1:
                    for i in range(sub.shape[1]):
                        out[mask, i] = rank_1d(sub[:, i])
                else:
                    out[mask] = rank_1d(sub)
        else:
            if arr.ndim > 1:
                for i in range(arr.shape[1]):
                    out[:, i] = rank_1d(arr[:, i])
            else:
                out = rank_1d(arr)
        
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._rank_array(X_out, keys)
        
        if self.target in ("y", "both"):
            y_out = self._rank_array(y_out, keys)
        
        return X_out, y_out
