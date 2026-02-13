"""
XGBoost feature importance extractor.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import polars as pl

from ..base import BaseImportanceExtractor


class XGBoostImportanceExtractor(BaseImportanceExtractor):
    def __init__(self, importance_type: str = "gain", normalize: bool = True):
        self.importance_type = importance_type
        self.normalize = normalize

    def extract(self, model: Any, feature_names: List[str]) -> Tuple[np.ndarray, pl.DataFrame]:
        importance = np.zeros(len(feature_names), dtype=np.float32)
        if hasattr(model, "get_score"):
            score = model.get_score(importance_type=self.importance_type)
            for idx, name in enumerate(feature_names):
                importance[idx] = float(score.get(name, 0.0))

        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)

        df = pl.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort("importance", descending=True)
        return importance, df
