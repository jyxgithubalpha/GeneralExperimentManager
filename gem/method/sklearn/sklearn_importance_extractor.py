"""
Sklearn importance extractor for linear and tree models.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import polars as pl

from ..base import BaseImportanceExtractor


class SklearnImportanceExtractor(BaseImportanceExtractor):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def extract(self, model: Any, feature_names: List[str]) -> Tuple[np.ndarray, pl.DataFrame]:
        importance = self._extract_raw_importance(model, len(feature_names))
        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)
        df = pl.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort("importance", descending=True)
        return importance, df

    @staticmethod
    def _extract_raw_importance(model: Any, n_features: int) -> np.ndarray:
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 1:
                importance = np.abs(coef)
            else:
                importance = np.mean(np.abs(coef), axis=0)
            if importance.shape[0] != n_features:
                return np.zeros(n_features, dtype=np.float32)
            return importance.astype(np.float32)

        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=np.float32)
            if importance.shape[0] != n_features:
                return np.zeros(n_features, dtype=np.float32)
            return importance

        return np.zeros(n_features, dtype=np.float32)
