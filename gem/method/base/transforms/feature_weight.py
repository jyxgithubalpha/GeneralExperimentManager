"""
FeatureWeightTransform - Feature weighting transform.
"""


from typing import Literal, Optional, Tuple, Union

import numpy as np

from ...method_dataclasses import TransformState
from .base import BaseTransform


class FeatureWeightTransform(BaseTransform):
    """
    Feature weighting transform.
    
    Gets feature_weights from context and applies weighting to X.
    Supports multiple weighting methods:
    - multiply: X * weights
    - sqrt_multiply: X * sqrt(weights) - softer weighting
    - rank_weight: X * rank_normalized_weights - rank-based weighting
    - softmax: X * softmax(weights / temperature)
    - select_topk: Keep only top-k important features
    
    Expected keys in context:
    - feature_weights: np.ndarray - Feature weight vector
    """
    
    def __init__(
        self,
        method: Literal["multiply", "sqrt_multiply", "rank_weight", "softmax", "select_topk"] = "multiply",
        topk: Optional[int] = None,
        temperature: float = 1.0,
        min_weight: float = 0.0,
        normalize: bool = False,
        fallback: Literal["uniform", "ones", "skip"] = "ones",
        context_key: str = "feature_weights",
    ):
        """
        Args:
            method: Weighting method
                - multiply: X * weights
                - sqrt_multiply: X * sqrt(weights) - softer weighting
                - rank_weight: X * rank_normalized_weights
                - softmax: X * softmax(weights / temperature)
                - select_topk: Keep only top-k important features
            topk: Number of top-k features to keep (only for select_topk)
            temperature: softmax temperature (only for softmax)
            min_weight: Minimum weight value (for clipping)
            normalize: Whether to normalize weights to sum to n_features
            fallback: Handling method when no feature_weights in context
                - uniform: uniform weights (1/n_features)
                - ones: all 1 weights (unchanged)
                - skip: skip transform
            context_key: Key to read weights from context
        """
        super().__init__()
        self.method = method
        self.topk = topk
        self.temperature = temperature
        self.min_weight = min_weight
        self.normalize = normalize
        self.fallback = fallback
        self.context_key = context_key
        self._weights: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "FeatureWeightTransform":
        n_features = X.shape[1]
        
        raw_weights = self.get_context_value(self.context_key)
        
        if raw_weights is None:
            raw_weights = self._get_fallback_weights(n_features)
        else:
            raw_weights = np.array(raw_weights, dtype=np.float32)
        
        if self.method == "multiply":
            self._weights = self._apply_constraints(raw_weights, n_features)
        elif self.method == "sqrt_multiply":
            sqrt_weights = np.sqrt(np.maximum(raw_weights, 0))
            self._weights = self._apply_constraints(sqrt_weights, n_features)
        elif self.method == "rank_weight":
            ranks = np.argsort(np.argsort(raw_weights)).astype(np.float32)
            rank_weights = (ranks + 1) / n_features  # normalize to [1/n, 1]
            self._weights = self._apply_constraints(rank_weights, n_features)
        elif self.method == "softmax":
            exp_w = np.exp((raw_weights - raw_weights.max()) / self.temperature)
            softmax_weights = exp_w / exp_w.sum()
            self._weights = self._apply_constraints(softmax_weights * n_features, n_features)
        elif self.method == "select_topk":
            k = self.topk or n_features
            k = min(k, n_features)
            self._selected_indices = np.argsort(raw_weights)[-k:]
            self._weights = None
        
        self._state = TransformState(stats={
            "weights": self._weights,
            "selected_indices": self._selected_indices,
            "method": self.method,
        })
        return self
    
    def _get_fallback_weights(self, n_features: int) -> Optional[np.ndarray]:
        if self.fallback == "uniform":
            return np.full(n_features, 1.0 / n_features, dtype=np.float32)
        elif self.fallback == "ones":
            return np.ones(n_features, dtype=np.float32)
        else:
            return np.ones(n_features, dtype=np.float32)
    
    def _apply_constraints(self, weights: np.ndarray, n_features: int) -> np.ndarray:
        """Apply min_weight clipping and optional normalization."""
        w = weights.copy()
        if self.min_weight > 0:
            w = np.maximum(w, self.min_weight)
        if self.normalize and w.sum() > 0:
            w = w * n_features / w.sum()
        return w
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.fallback == "skip" and self._weights is None and self._selected_indices is None:
            return X, y
        
        X_out = X.copy()
        
        if self.method == "select_topk" and self._selected_indices is not None:
            X_out = X_out[:, self._selected_indices]
        elif self._weights is not None:
            X_out = X_out * self._weights
        
        return X_out, y
    
    @property
    def selected_feature_indices(self) -> Optional[np.ndarray]:
        """Get selected feature indices (select_topk mode)."""
        return self._selected_indices
