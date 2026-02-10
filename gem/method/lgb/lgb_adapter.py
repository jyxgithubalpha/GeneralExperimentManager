"""
LightGBMAdapter - LightGBM 数据适配器

将 SplitView/RayDataBundle 转换为 lgb.Dataset

支持:
- SplitView -> lgb.Dataset (直接转换)
- RayDataBundle -> lgb.Dataset (从 numpy 转换)
- 样本权重支持
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import BaseAdapter
from ..method_dataclasses import RayDataBundle, RayDataViews
from ...data.data_dataclasses import ProcessedViews, SplitView


class LightGBMAdapter(BaseAdapter):
    """
    LightGBM 数据集适配器
    
    将 SplitView/RayDataBundle 转换为 lgb.Dataset
    支持样本权重和特征名传递
    """
    
    def __init__(
        self,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        free_raw_data: bool = True,
    ):
        """
        Args:
            feature_name: 特征名设置 ("auto" 或特征名列表)
            categorical_feature: 类别特征设置
            free_raw_data: 是否释放原始数据
        """
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.free_raw_data = free_raw_data
    
    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        """
        将 SplitView 转换为 lgb.Dataset
        
        Args:
            view: SplitView 实例
            reference: 参考 dataset (用于验证集)
            weight: 可选的样本权重
            **kwargs: 传递给 lgb.Dataset 的额外参数
            
        Returns:
            lgb.Dataset 实例
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMAdapter")
        
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        
        # 确定 feature_name
        feature_name = self.feature_name
        if feature_name == "auto" and view.feature_names:
            feature_name = view.feature_names
        
        dataset = lgb.Dataset(
            data=view.X,
            label=y,
            weight=weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs
        )
        
        return dataset
    
    def from_ray_bundle(
        self,
        bundle: RayDataBundle,
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        从 RayDataBundle 转换为 lgb.Dataset
        
        Args:
            bundle: RayDataBundle 实例
            reference: 参考 dataset
            **kwargs: 传递给 lgb.Dataset 的额外参数
            
        Returns:
            lgb.Dataset 实例
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMAdapter")
        
        y = bundle.y.ravel() if bundle.y.ndim > 1 else bundle.y
        
        # 确定 feature_name
        feature_name = self.feature_name
        if feature_name == "auto" and bundle.feature_names:
            feature_name = bundle.feature_names
        
        dataset = lgb.Dataset(
            data=bundle.X,
            label=y,
            weight=bundle.sample_weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs
        )
        
        return dataset
    
    def from_numpy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        直接从 numpy 数组创建 lgb.Dataset
        
        Args:
            X: 特征矩阵
            y: 标签
            weight: 可选的样本权重
            feature_names: 可选的特征名列表
            reference: 参考 dataset
            **kwargs: 传递给 lgb.Dataset 的额外参数
            
        Returns:
            lgb.Dataset 实例
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMAdapter")
        
        y_flat = y.ravel() if y.ndim > 1 else y
        
        feature_name = self.feature_name
        if feature_name == "auto" and feature_names:
            feature_name = feature_names
        
        dataset = lgb.Dataset(
            data=X,
            label=y_flat,
            weight=weight,
            feature_name=feature_name,
            categorical_feature=self.categorical_feature,
            reference=reference,
            free_raw_data=self.free_raw_data,
            **kwargs
        )
        
        return dataset
    
    def to_train_val_test(
        self,
        views: ProcessedViews,
        weights: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """
        将 ProcessedViews 转换为 (train, val, test) lgb.Dataset 三元组
        
        Args:
            views: ProcessedViews 实例
            weights: 可选的样本权重字典 {"train": ..., "val": ..., "test": ...}
            **kwargs: 传递给 lgb.Dataset 的额外参数
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        weights = weights or {}
        
        dtrain = self.to_dataset(views.train, weight=weights.get("train"), **kwargs)
        dval = self.to_dataset(views.val, reference=dtrain, weight=weights.get("val"), **kwargs)
        dtest = self.to_dataset(views.test, reference=dtrain, weight=weights.get("test"), **kwargs)
        
        return dtrain, dval, dtest
    
    def from_ray_views(
        self,
        ray_views: RayDataViews,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """
        从 RayDataViews 转换为 (train, val, test) lgb.Dataset 三元组
        
        Args:
            ray_views: RayDataViews 实例
            **kwargs: 传递给 lgb.Dataset 的额外参数
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        dtrain = self.from_ray_bundle(ray_views.train, **kwargs)
        dval = self.from_ray_bundle(ray_views.val, reference=dtrain, **kwargs)
        dtest = self.from_ray_bundle(ray_views.test, reference=dtrain, **kwargs)
        
        return dtrain, dval, dtest
    
    def create_datasets_dict(
        self,
        views: ProcessedViews,
        weights: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建 datasets 字典
        
        Args:
            views: ProcessedViews 实例
            weights: 可选的样本权重字典
            **kwargs: 传递给 lgb.Dataset 的额外参数
            
        Returns:
            {"train": lgb.Dataset, "val": lgb.Dataset, "test": lgb.Dataset}
        """
        dtrain, dval, dtest = self.to_train_val_test(views, weights, **kwargs)
        return {"train": dtrain, "val": dval, "test": dtest}
    
    def create_datasets_from_ray(
        self,
        ray_views: RayDataViews,
        **kwargs
    ) -> Dict[str, Any]:
        """
        从 RayDataViews 创建 datasets 字典
        
        Args:
            ray_views: RayDataViews 实例
            **kwargs: 传递给 lgb.Dataset 的额外参数
            
        Returns:
            {"train": lgb.Dataset, "val": lgb.Dataset, "test": lgb.Dataset}
        """
        dtrain, dval, dtest = self.from_ray_views(ray_views, **kwargs)
        return {"train": dtrain, "val": dval, "test": dtest}
