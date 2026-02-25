"""
StandardizeTransform - Standardization transform.
"""


from typing import Literal, Optional, Tuple

import numpy as np

from ...method_dataclasses import TransformState
from .base import BaseTransform


class StandardizeTransform(BaseTransform):
    """Standardization."""
    
    def __init__(
        self,
        target: Literal["X", "y", "both"] = "X",
        eps: float = 1e-8,
        per_date: bool = True,
    ):
        super().__init__()
        self.target = target
        self.eps = eps
        self.per_date = per_date
        self._mean_X: Optional[np.ndarray] = None
        self._std_X: Optional[np.ndarray] = None
        self._mean_y: Optional[np.ndarray] = None
        self._std_y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "StandardizeTransform":
        if not self.per_date or keys is None:
            self._mean_X = np.nanmean(X, axis=0)
            self._std_X = np.nanstd(X, axis=0)
            if y.ndim > 1:
                self._mean_y = np.nanmean(y, axis=0)
                self._std_y = np.nanstd(y, axis=0)
            else:
                self._mean_y = np.array([np.nanmean(y)])
                self._std_y = np.array([np.nanstd(y)])
        
        self._state = TransformState(stats={
            "mean_X": self._mean_X,
            "std_X": self._std_X,
            "mean_y": self._mean_y,
            "std_y": self._std_y,
            "per_date": self.per_date,
        })
        return self
    
    def _standardize_array(self, arr: np.ndarray, mean: np.ndarray, std: np.ndarray, keys: Optional[np.ndarray]) -> np.ndarray:
        out = arr.copy()
        if self.per_date and keys is not None:
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = out[mask]
                m = np.nanmean(sub, axis=0)
                s = np.nanstd(sub, axis=0) + self.eps
                out[mask] = (sub - m) / s
        else:
            out = (out - mean) / (std + self.eps)
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._standardize_array(X_out, self._mean_X, self._std_X, keys)
        
        if self.target in ("y", "both"):
            y_out = self._standardize_array(y_out, self._mean_y, self._std_y, keys)
        
        return X_out, y_out
    
    def inverse_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if not self.per_date or keys is None:
            if self.target in ("X", "both"):
                X_out = X_out * (self._std_X + self.eps) + self._mean_X
            if self.target in ("y", "both"):
                y_out = y_out * (self._std_y + self.eps) + self._mean_y
        
        return X_out, y_out
