"""
CatBoost feature importance extractor.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import polars as pl

from ..base import BaseImportanceExtractor


class CatBoostImportanceExtractor(BaseImportanceExtractor):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def extract(self, model: Any, feature_names: List[str]) -> Tuple[np.ndarray, pl.DataFrame]:
        importance = np.zeros(len(feature_names), dtype=np.float32)
        if hasattr(model, "get_feature_importance"):
            importance = np.asarray(model.get_feature_importance(), dtype=np.float32)
            if importance.shape[0] != len(feature_names):
                importance = np.zeros(len(feature_names), dtype=np.float32)

        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)

        df = pl.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort("importance", descending=True)
        return importance, df
