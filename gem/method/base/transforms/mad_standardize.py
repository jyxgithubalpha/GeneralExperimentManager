"""
MADStandardizeTransform - Robust standardization using Median and MAD.
"""


from typing import Literal, Optional, Tuple

import numpy as np

from ...method_dataclasses import TransformState
from .base import BaseTransform


class MADStandardizeTransform(BaseTransform):
    """
    Robust standardization using Median and MAD (Median Absolute Deviation).
    
    Formula:
        MAD = median(|X - median|)
        scale = 1.4826 * MAD (fallback to std if scale < eps)
        X_standardized = (X - median) / scale
    
    The constant 1.4826 makes MAD consistent with standard deviation for normal distributions.
    """
    
    def __init__(
        self,
        target: Literal["X", "y", "both"] = "X",
        eps: float = 1e-6,
        per_date: bool = True,
    ):
        super().__init__()
        self.target = target
        self.eps = eps
        self.per_date = per_date
        self._median_X: Optional[np.ndarray] = None
        self._scale_X: Optional[np.ndarray] = None
        self._median_y: Optional[np.ndarray] = None
        self._scale_y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "MADStandardizeTransform":
        if not self.per_date or keys is None:
            self._median_X = np.nanmedian(X, axis=0)
            mad_X = np.nanmedian(np.abs(X - self._median_X), axis=0)
            self._scale_X = 1.4826 * mad_X
            std_fallback_X = np.nanstd(X, axis=0)
            self._scale_X = np.where(self._scale_X < self.eps, std_fallback_X + self.eps, self._scale_X)
            
            if y.ndim > 1:
                self._median_y = np.nanmedian(y, axis=0)
                mad_y = np.nanmedian(np.abs(y - self._median_y), axis=0)
                self._scale_y = 1.4826 * mad_y
                std_fallback_y = np.nanstd(y, axis=0)
                self._scale_y = np.where(self._scale_y < self.eps, std_fallback_y + self.eps, self._scale_y)
            else:
                self._median_y = np.array([np.nanmedian(y)])
                mad_y = np.nanmedian(np.abs(y - self._median_y[0]))
                scale_y = 1.4826 * mad_y
                std_fallback_y = np.nanstd(y)
                self._scale_y = np.array([scale_y if scale_y >= self.eps else std_fallback_y + self.eps])
        
        self._state = TransformState(stats={
            "median_X": self._median_X,
            "scale_X": self._scale_X,
            "median_y": self._median_y,
            "scale_y": self._scale_y,
            "per_date": self.per_date,
        })
        return self
    
    def _mad_standardize_array(
        self,
        arr: np.ndarray,
        median: Optional[np.ndarray],
        scale: Optional[np.ndarray],
        keys: Optional[np.ndarray],
    ) -> np.ndarray:
        out = arr.copy()
        if self.per_date and keys is not None:
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = out[mask]
                
                if sub.ndim > 1:
                    med = np.nanmedian(sub, axis=0)
                    mad = np.nanmedian(np.abs(sub - med), axis=0)
                    sc = 1.4826 * mad
                    std_fallback = np.nanstd(sub, axis=0)
                    sc = np.where(sc < self.eps, std_fallback + self.eps, sc)
                    out[mask] = (sub - med) / sc
                else:
                    med = np.nanmedian(sub)
                    mad = np.nanmedian(np.abs(sub - med))
                    sc = 1.4826 * mad
                    if sc < self.eps:
                        sc = np.nanstd(sub) + self.eps
                    out[mask] = (sub - med) / sc
        else:
            out = (out - median) / scale
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._mad_standardize_array(X_out, self._median_X, self._scale_X, keys)
        
        if self.target in ("y", "both"):
            y_out = self._mad_standardize_array(y_out, self._median_y, self._scale_y, keys)
        
        return X_out, y_out
