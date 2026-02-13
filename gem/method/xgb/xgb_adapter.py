"""
XGBoost adapter - convert SplitView/RayDataBundle to xgboost.DMatrix.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import BaseAdapter
from ..method_dataclasses import RayDataBundle
from ...data.data_dataclasses import SplitView


class XGBoostAdapter(BaseAdapter):
    def __init__(self, enable_categorical: bool = False):
        self.enable_categorical = enable_categorical

    @staticmethod
    def _import_xgboost():
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostAdapter") from exc
        return xgb

    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        xgb = self._import_xgboost()
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        return xgb.DMatrix(
            data=view.X,
            label=y,
            weight=weight,
            feature_names=view.feature_names,
            enable_categorical=self.enable_categorical,
            **kwargs,
        )

    def from_ray_bundle(
        self,
        bundle: RayDataBundle,
        reference: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        xgb = self._import_xgboost()
        y = bundle.y.ravel() if bundle.y.ndim > 1 else bundle.y
        return xgb.DMatrix(
            data=bundle.X,
            label=y,
            weight=bundle.sample_weight,
            feature_names=bundle.feature_names,
            enable_categorical=self.enable_categorical,
            **kwargs,
        )
