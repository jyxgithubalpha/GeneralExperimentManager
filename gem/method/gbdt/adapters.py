"""
GBDT Adapters - Data adapters for LightGBM, XGBoost, CatBoost.
"""


from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import BaseAdapter
from ..method_dataclasses import RayDataBundle, RayDataViews
from ...data.data_dataclasses import ProcessedViews, SplitView


class LightGBMAdapter(BaseAdapter):
    """LightGBM data adapter - converts SplitView/RayDataBundle to lgb.Dataset."""
    
    def __init__(
        self,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        free_raw_data: bool = True,
    ):
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.free_raw_data = free_raw_data
    
    @staticmethod
    def _import_lgb():
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMAdapter")
        return lgb
    
    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        lgb = self._import_lgb()
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        feature_name = self.feature_name
        if feature_name == "auto" and view.feature_names:
            feature_name = view.feature_names
        
        return lgb.Dataset(
            data=view.X,
            label=y,
            weight=weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs
        )
    
    def from_ray_bundle(
        self,
        bundle: RayDataBundle,
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        lgb = self._import_lgb()
        y = bundle.y.ravel() if bundle.y.ndim > 1 else bundle.y
        feature_name = self.feature_name
        if feature_name == "auto" and bundle.feature_names:
            feature_name = bundle.feature_names
        
        return lgb.Dataset(
            data=bundle.X,
            label=y,
            weight=bundle.sample_weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs
        )
    
    def from_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        lgb = self._import_lgb()
        y_flat = y.ravel() if y.ndim > 1 else y
        feature_name = self.feature_name
        if feature_name == "auto" and feature_names:
            feature_name = feature_names
        
        return lgb.Dataset(
            data=X,
            label=y_flat,
            weight=weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs
        )
    
    def to_train_val_test(
        self,
        views: ProcessedViews,
        weights: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        weights = weights or {}
        dtrain = self.to_dataset(views.train, weight=weights.get("train"), **kwargs)
        dval = self.to_dataset(views.val, reference=dtrain, weight=weights.get("val"), **kwargs)
        dtest = self.to_dataset(views.test, reference=dtrain, weight=weights.get("test"), **kwargs)
        return dtrain, dval, dtest
    
    def create_datasets_dict(
        self,
        views: ProcessedViews,
        weights: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        dtrain, dval, dtest = self.to_train_val_test(views, weights, **kwargs)
        return {"train": dtrain, "val": dval, "test": dtest}


class XGBoostAdapter(BaseAdapter):
    """XGBoost data adapter - converts SplitView/RayDataBundle to xgb.DMatrix."""
    
    def __init__(self, enable_categorical: bool = False):
        self.enable_categorical = enable_categorical

    @staticmethod
    def _import_xgb():
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
        xgb = self._import_xgb()
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
        xgb = self._import_xgb()
        y = bundle.y.ravel() if bundle.y.ndim > 1 else bundle.y
        return xgb.DMatrix(
            data=bundle.X,
            label=y,
            weight=bundle.sample_weight,
            feature_names=bundle.feature_names,
            enable_categorical=self.enable_categorical,
            **kwargs,
        )


class CatBoostAdapter(BaseAdapter):
    """CatBoost data adapter - converts SplitView/RayDataBundle to catboost.Pool."""
    
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
