"""
MinSampleFilterTransform - Filter out date groups with too few samples.
"""


from typing import Optional, Tuple

import numpy as np

from ...method_dataclasses import TransformState
from .base import BaseTransform


class MinSampleFilterTransform(BaseTransform):
    """
    Filter out date groups with sample count below threshold.
    
    Sets features and labels to NaN for date groups with sample count < min_samples.
    These NaN values should be handled by subsequent transforms or will be skipped
    during evaluation.
    """
    
    def __init__(
        self,
        min_samples: int = 5,
    ):
        super().__init__()
        self.min_samples = min_samples
        self._filtered_dates: list = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "MinSampleFilterTransform":
        self._filtered_dates = []
        
        if keys is not None:
            unique_keys, counts = np.unique(keys, return_counts=True)
            for key, count in zip(unique_keys, counts):
                if count < self.min_samples:
                    self._filtered_dates.append(key)
        
        self._state = TransformState(stats={
            "min_samples": self.min_samples,
            "filtered_dates": self._filtered_dates,
            "n_filtered": len(self._filtered_dates),
        })
        return self
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if keys is not None:
            unique_keys, counts = np.unique(keys, return_counts=True)
            for key, count in zip(unique_keys, counts):
                if count < self.min_samples:
                    mask = keys == key
                    X_out[mask] = np.nan
                    y_out[mask] = np.nan
        
        return X_out, y_out
