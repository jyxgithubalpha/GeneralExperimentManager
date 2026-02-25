"""
GBDT Importance Extractor - Unified feature importance extraction for GBDT models.
"""


from typing import Any, Callable, List, Tuple

import numpy as np
import polars as pl

from ..base.base_importance_extractor import BaseImportanceExtractor


class GBDTImportanceExtractor(BaseImportanceExtractor):
    """
    Unified GBDT feature importance extractor.
    
    Supports LightGBM, XGBoost, CatBoost through configurable extract function.
    """
    
    def __init__(
        self,
        framework: str = "auto",
        importance_type: str = "gain",
        normalize: bool = True,
    ):
        """
        Args:
            framework: "lightgbm", "xgboost", "catboost", or "auto" (detect)
            importance_type: Type of importance ("gain", "split", etc.)
            normalize: Whether to normalize importance to sum to 1
        """
        self.framework = framework
        self.importance_type = importance_type
        self.normalize = normalize
    
    def extract(self, model: Any, feature_names: List[str]) -> Tuple[np.ndarray, pl.DataFrame]:
        n_features = len(feature_names)
        
        framework = self._detect_framework(model) if self.framework == "auto" else self.framework
        importance = self._extract_by_framework(model, feature_names, framework)
        
        if importance.shape[0] != n_features:
            importance = np.zeros(n_features, dtype=np.float32)
        
        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)
        
        df = pl.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort("importance", descending=True)
        
        return importance, df
    
    def _detect_framework(self, model: Any) -> str:
        """Auto-detect model framework."""
        model_module = model.__class__.__module__
        model_name = model.__class__.__name__
        
        if "lightgbm" in model_module:
            return "lightgbm"
        if "xgboost" in model_module:
            return "xgboost"
        if "catboost" in model_module:
            return "catboost"
        
        if hasattr(model, "feature_importance"):
            if hasattr(model, "best_iteration"):
                return "lightgbm"
        if hasattr(model, "get_score"):
            return "xgboost"
        if hasattr(model, "get_feature_importance"):
            return "catboost"
        
        return "unknown"
    
    def _extract_by_framework(
        self,
        model: Any,
        feature_names: List[str],
        framework: str,
    ) -> np.ndarray:
        """Extract importance based on framework."""
        n_features = len(feature_names)
        
        if framework == "lightgbm":
            return self._extract_lightgbm(model, n_features)
        elif framework == "xgboost":
            return self._extract_xgboost(model, feature_names)
        elif framework == "catboost":
            return self._extract_catboost(model, n_features)
        else:
            return np.zeros(n_features, dtype=np.float32)
    
    def _extract_lightgbm(self, model: Any, n_features: int) -> np.ndarray:
        """Extract LightGBM importance."""
        try:
            importance = model.feature_importance(importance_type=self.importance_type)
            return importance.astype(np.float32)
        except Exception:
            return np.zeros(n_features, dtype=np.float32)
    
    def _extract_xgboost(self, model: Any, feature_names: List[str]) -> np.ndarray:
        """Extract XGBoost importance."""
        n_features = len(feature_names)
        try:
            score = model.get_score(importance_type=self.importance_type)
            importance = np.zeros(n_features, dtype=np.float32)
            for idx, name in enumerate(feature_names):
                importance[idx] = float(score.get(name, 0.0))
            return importance
        except Exception:
            return np.zeros(n_features, dtype=np.float32)
    
    def _extract_catboost(self, model: Any, n_features: int) -> np.ndarray:
        """Extract CatBoost importance."""
        try:
            importance = np.asarray(model.get_feature_importance(), dtype=np.float32)
            return importance
        except Exception:
            return np.zeros(n_features, dtype=np.float32)


def create_importance_extractor(
    framework: str,
    importance_type: str = "gain",
    normalize: bool = True,
) -> GBDTImportanceExtractor:
    """
    Factory function to create importance extractor.
    
    Args:
        framework: "lightgbm", "xgboost", "catboost", or "auto"
        importance_type: Type of importance
        normalize: Whether to normalize
        
    Returns:
        GBDTImportanceExtractor instance
    """
    return GBDTImportanceExtractor(
        framework=framework,
        importance_type=importance_type,
        normalize=normalize,
    )
