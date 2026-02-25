"""
FillNaNTransform - Fill missing values transform.
"""


from typing import Literal, Optional, Tuple

import numpy as np

from ...method_dataclasses import TransformState
from .base import BaseTransform


class FillNaNTransform(BaseTransform):
    """Fill missing values."""
    
    def __init__(
        self,
        value: float = 0.0,
        method: Literal["constant", "mean", "median"] = "constant",
        target: Literal["X", "y", "both"] = "both",
        per_date: bool = False,
    ):
        super().__init__()
        self.value = value
        self.method = method
        self.target = target
        self.per_date = per_date
        self._fill_values_X: Optional[np.ndarray] = None
        self._fill_values_y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "FillNaNTransform":
        if self.method == "constant":
            self._fill_values_X = np.full(X.shape[1], self.value)
            self._fill_values_y = np.full(y.shape[1] if y.ndim > 1 else 1, self.value)
        elif self.method == "mean":
            self._fill_values_X = np.nanmean(X, axis=0)
            self._fill_values_y = np.nanmean(y, axis=0) if y.ndim > 1 else np.array([np.nanmean(y)])
        elif self.method == "median":
            self._fill_values_X = np.nanmedian(X, axis=0)
            self._fill_values_y = np.nanmedian(y, axis=0) if y.ndim > 1 else np.array([np.nanmedian(y)])
        
        self._state = TransformState(stats={
            "fill_values_X": self._fill_values_X,
            "fill_values_y": self._fill_values_y,
            "per_date": self.per_date,
        })
        return self
    
    def _fill_array(self, arr: np.ndarray, fill_values: np.ndarray, keys: Optional[np.ndarray]) -> np.ndarray:
        out = arr.copy()
        if self.per_date and keys is not None and self.method in ("mean", "median"):
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = out[mask]
                if self.method == "mean":
                    fill_vals = np.nanmean(sub, axis=0)
                else:
                    fill_vals = np.nanmedian(sub, axis=0)
                nan_mask = np.isnan(sub)
                if nan_mask.any():
                    if sub.ndim > 1:
                        for i in range(sub.shape[1]):
                            col_nan = nan_mask[:, i]
                            sub[col_nan, i] = fill_vals[i] if not np.isnan(fill_vals[i]) else self.value
                    else:
                        sub[nan_mask] = fill_vals if not np.isnan(fill_vals) else self.value
                out[mask] = sub
        else:
            if out.ndim > 1:
                for i in range(out.shape[1]):
                    col_nan = np.isnan(out[:, i])
                    out[col_nan, i] = fill_values[i]
            else:
                nan_mask = np.isnan(out)
                out[nan_mask] = fill_values[0]
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._fill_array(X_out, self._fill_values_X, keys)
        
        if self.target in ("y", "both"):
            y_out = self._fill_array(y_out, self._fill_values_y, keys)
        
        return X_out, y_out
