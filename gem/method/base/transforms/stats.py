"""
StatsCalculator - Statistics calculator for transforms.
"""


from typing import Optional

import numpy as np

from ...method_dataclasses import TransformStats


class StatsCalculator:
    """
    Statistics calculator.
    
    Computes X and y statistics from train/val data:
    - Mean
    - Standard deviation
    - Quantiles
    - Median
    """
    
    @staticmethod
    def compute(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        use_combined: bool = False,
    ) -> TransformStats:
        """
        Compute statistics.
        
        Args:
            X_train: Training set features
            y_train: Training set labels
            X_val: Validation set features (optional)
            y_val: Validation set labels (optional)
            lower_quantile: Lower quantile
            upper_quantile: Upper quantile
            use_combined: Whether to combine train and val for computation
            
        Returns:
            TransformStats
        """
        if use_combined and X_val is not None:
            X = np.vstack([X_train, X_val])
            y = np.concatenate([y_train.ravel(), y_val.ravel()])
        else:
            X = X_train
            y = y_train.ravel() if y_train.ndim > 1 else y_train
        
        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)
        X_lower = np.nanquantile(X, lower_quantile, axis=0)
        X_upper = np.nanquantile(X, upper_quantile, axis=0)
        X_median = np.nanmedian(X, axis=0)
        
        y_mean = np.nanmean(y)
        y_std = np.nanstd(y)
        y_lower = np.nanquantile(y, lower_quantile)
        y_upper = np.nanquantile(y, upper_quantile)
        y_median = np.nanmedian(y)
        
        return TransformStats(
            X_mean=X_mean,
            X_std=X_std,
            X_lower_quantile=X_lower,
            X_upper_quantile=X_upper,
            X_median=X_median,
            y_mean=np.array([y_mean]),
            y_std=np.array([y_std]),
            y_lower_quantile=np.array([y_lower]),
            y_upper_quantile=np.array([y_upper]),
            y_median=np.array([y_median]),
        )
