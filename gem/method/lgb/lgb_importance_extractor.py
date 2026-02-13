"""
LightGBM feature importance extractor.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import polars as pl

from ..base import BaseImportanceExtractor


class LightGBMImportanceExtractor(BaseImportanceExtractor):
    """Extract aligned feature importance vector and table from a LightGBM model."""

    def __init__(self, importance_type: str = "gain", normalize: bool = True):
        self.importance_type = importance_type
        self.normalize = normalize

    def extract(self, model: Any, feature_names: List[str]) -> Tuple[np.ndarray, pl.DataFrame]:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm is required for LightGBMImportanceExtractor") from exc

        if not isinstance(model, lgb.Booster):
            importance = np.zeros(len(feature_names), dtype=np.float32)
            return pluck_importance(feature_names, importance)

        importance = model.feature_importance(importance_type=self.importance_type).astype(np.float32)
        if importance.shape[0] != len(feature_names):
            raise ValueError(
                "Feature importance length mismatch: "
                f"{importance.shape[0]} vs {len(feature_names)}"
            )

        if self.normalize and float(np.sum(importance)) > 0:
            importance = importance / np.sum(importance)

        return pluck_importance(feature_names, importance)


def pluck_importance(feature_names: List[str], importance: np.ndarray) -> Tuple[np.ndarray, pl.DataFrame]:
    df = pl.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort("importance", descending=True)
    return importance, df
