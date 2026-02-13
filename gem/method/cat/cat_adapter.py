"""
CatBoost adapter - convert SplitView/RayDataBundle to catboost.Pool.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..base import BaseAdapter
from ..method_dataclasses import RayDataBundle
from ...data.data_dataclasses import SplitView


class CatBoostAdapter(BaseAdapter):
    def __init__(self, cat_features: Optional[List[int]] = None):
        self.cat_features = cat_features or []

    @staticmethod
    def _import_catboost():
        try:
            from catboost import Pool
        except ImportError as exc:
            raise ImportError("catboost is required for CatBoostAdapter") from exc
        return Pool

    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        Pool = self._import_catboost()
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        return Pool(
            data=view.X,
            label=y,
            weight=weight,
            feature_names=view.feature_names,
            cat_features=self.cat_features or None,
            **kwargs,
        )

    def from_ray_bundle(
        self,
        bundle: RayDataBundle,
        reference: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        Pool = self._import_catboost()
        y = bundle.y.ravel() if bundle.y.ndim > 1 else bundle.y
        return Pool(
            data=bundle.X,
            label=y,
            weight=bundle.sample_weight,
            feature_names=bundle.feature_names,
            cat_features=self.cat_features or None,
            **kwargs,
        )
