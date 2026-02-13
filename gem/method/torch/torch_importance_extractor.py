"""
Torch feature importance extractor (weight-norm proxy).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import polars as pl

from ..base import BaseImportanceExtractor


class TorchImportanceExtractor(BaseImportanceExtractor):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def extract(self, model: Any, feature_names: List[str]) -> Tuple[np.ndarray, pl.DataFrame]:
        if hasattr(model, "model"):
            base = model.model
        else:
            base = model

        importance = self._extract_from_model(base, len(feature_names))
        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)

        df = pl.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort("importance", descending=True)
        return importance, df

    @staticmethod
    def _extract_from_model(model: Any, n_features: int) -> np.ndarray:
        if hasattr(model, "feature_importance"):
            importance = model.feature_importance().detach().cpu().numpy().astype(np.float32)
            if importance.shape[0] == n_features:
                return importance
        return np.ones(n_features, dtype=np.float32)
