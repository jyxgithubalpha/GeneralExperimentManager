"""
WinsorizeTransform - Winsorization processing transform.
"""


from typing import Literal, Optional, Tuple

import numpy as np

from ...method_dataclasses import TransformState
from .base import BaseTransform


class WinsorizeTransform(BaseTransform):
    """Winsorization processing."""
    
    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 0.99,
        target: Literal["X", "y", "both"] = "X",
        per_date: bool = True,
    ):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.target = target
        self.per_date = per_date
        self._lower_bounds_X: Optional[np.ndarray] = None
        self._upper_bounds_X: Optional[np.ndarray] = None
        self._lower_bounds_y: Optional[np.ndarray] = None
        self._upper_bounds_y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "WinsorizeTransform":
        if not self.per_date or keys is None:
            self._lower_bounds_X = np.nanquantile(X, self.lower, axis=0)
            self._upper_bounds_X = np.nanquantile(X, self.upper, axis=0)
            if y.ndim > 1:
                self._lower_bounds_y = np.nanquantile(y, self.lower, axis=0)
                self._upper_bounds_y = np.nanquantile(y, self.upper, axis=0)
            else:
                self._lower_bounds_y = np.array([np.nanquantile(y, self.lower)])
                self._upper_bounds_y = np.array([np.nanquantile(y, self.upper)])
        
        self._state = TransformState(stats={
            "lower_bounds_X": self._lower_bounds_X,
            "upper_bounds_X": self._upper_bounds_X,
            "lower_bounds_y": self._lower_bounds_y,
            "upper_bounds_y": self._upper_bounds_y,
            "per_date": self.per_date,
        })
        return self
    
    def _winsorize_array(self, arr: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray, keys: Optional[np.ndarray]) -> np.ndarray:
        out = arr.copy()
        if self.per_date and keys is not None:
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = out[mask]
                lb = np.nanquantile(sub, self.lower, axis=0)
                ub = np.nanquantile(sub, self.upper, axis=0)
                out[mask] = np.clip(sub, lb, ub)
        else:
            out = np.clip(out, lower_bounds, upper_bounds)
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._winsorize_array(X_out, self._lower_bounds_X, self._upper_bounds_X, keys)
        
        if self.target in ("y", "both"):
            y_out = self._winsorize_array(y_out, self._lower_bounds_y, self._upper_bounds_y, keys)
        
        return X_out, y_out
