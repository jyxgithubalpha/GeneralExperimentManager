"""
DatasetAdapter - 数据集适配器基类

将 ProcessedViews/SplitView/RayDataBundle 转换为各后端 dataset

流程:
1. pl.DataFrame -> numpy (在 SplitView/RayDataBundle 中完成)
2. numpy -> ray.data.Dataset (通过 RayDataBundle.to_ray_dataset())
3. numpy/ray.data -> 后端 dataset (如 lgb.Dataset)

支持的转换路径:
- SplitView -> BackendDataset (直接转换)
- RayDataBundle -> BackendDataset (从 numpy 转换)
- RayDataBundle -> ray.data.Dataset -> BackendDataset (分布式)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

from ...data.data_dataclasses import ProcessedViews, SplitView
from ..method_dataclasses import RayDataBundle, RayDataViews


class BaseAdapter(ABC):
    """
    数据集适配器基类
    
    负责将数据转换为特定后端的 dataset 格式
    """
    
    @abstractmethod
    def to_dataset(
        self,
        view: "SplitView",
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        将 SplitView 转换为后端 dataset
        
        Args:
            view: SplitView 实例
            reference: 参考 dataset (用于验证集/测试集)
            **kwargs: 后端特定参数
            
        Returns:
            后端 dataset 实例
        """
        pass
    
    @abstractmethod
    def from_ray_bundle(
        self,
        bundle: "RayDataBundle",
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        从 RayDataBundle 转换为后端 dataset
        
        Args:
            bundle: RayDataBundle 实例
            reference: 参考 dataset
            **kwargs: 后端特定参数
            
        Returns:
            后端 dataset 实例
        """
        pass
    
    def to_train_val_test(
        self,
        views: "ProcessedViews",
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """
        将 ProcessedViews 转换为 (train, val, test) 三元组
        
        Args:
            views: ProcessedViews 实例
            **kwargs: 后端特定参数
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        dtrain = self.to_dataset(views.train, **kwargs)
        dval = self.to_dataset(views.val, reference=dtrain, **kwargs)
        dtest = self.to_dataset(views.test, reference=dtrain, **kwargs)
        return dtrain, dval, dtest
    
    def from_ray_views(
        self,
        ray_views: "RayDataViews",
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """
        从 RayDataViews 转换为 (train, val, test) 三元组
        
        Args:
            ray_views: RayDataViews 实例
            **kwargs: 后端特定参数
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        dtrain = self.from_ray_bundle(ray_views.train, **kwargs)
        dval = self.from_ray_bundle(ray_views.val, reference=dtrain, **kwargs)
        dtest = self.from_ray_bundle(ray_views.test, reference=dtrain, **kwargs)
        return dtrain, dval, dtest
    
    def create_datasets_dict(
        self,
        views: "ProcessedViews",
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建 datasets 字典
        
        Args:
            views: ProcessedViews 实例
            **kwargs: 后端特定参数
            
        Returns:
            {"train": dataset, "val": dataset, "test": dataset}
        """
        dtrain, dval, dtest = self.to_train_val_test(views, **kwargs)
        return {"train": dtrain, "val": dval, "test": dtest}


class RayDataAdapter:
    """
    Ray Data 适配器
    
    负责 pl.DataFrame/SplitView -> numpy -> ray.data.Dataset 的转换
    """
    
    @staticmethod
    def split_view_to_bundle(
        view: "SplitView",
        sample_weight: Optional[np.ndarray] = None,
    ) -> RayDataBundle:
        """
        将 SplitView 转换为 RayDataBundle
        
        Args:
            view: SplitView 实例
            sample_weight: 可选的样本权重
            
        Returns:
            RayDataBundle
        """
        keys_arr = None
        if view.keys is not None:
            keys_arr = view.keys.to_numpy()
        
        return RayDataBundle(
            X=view.X,
            y=view.y,
            keys=keys_arr,
            sample_weight=sample_weight,
            feature_names=view.feature_names,
            label_names=view.label_names,
        )
    
    @staticmethod
    def views_to_ray_views(
        views: "ProcessedViews",
        sample_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> RayDataViews:
        """
        将 ProcessedViews 转换为 RayDataViews
        
        Args:
            views: ProcessedViews 实例
            sample_weights: 可选的样本权重字典 {"train": ..., "val": ..., "test": ...}
            
        Returns:
            RayDataViews
        """
        weights = sample_weights or {}
        
        return RayDataViews(
            train=RayDataAdapter.split_view_to_bundle(
                views.train, weights.get("train")
            ),
            val=RayDataAdapter.split_view_to_bundle(
                views.val, weights.get("val")
            ),
            test=RayDataAdapter.split_view_to_bundle(
                views.test, weights.get("test")
            ),
            transform_state=views.transform_state,
        )
    
    @staticmethod
    def to_ray_datasets(
        ray_views: "RayDataViews",
        include_weight: bool = True,
    ) -> Dict[str, Any]:
        """
        将 RayDataViews 转换为 ray.data.Dataset 字典
        
        Args:
            ray_views: RayDataViews 实例
            include_weight: 是否包含样本权重
            
        Returns:
            {"train": ray.data.Dataset, "val": ..., "test": ...}
        """
        return {
            "train": ray_views.train.to_ray_dataset(include_weight),
            "val": ray_views.val.to_ray_dataset(include_weight),
            "test": ray_views.test.to_ray_dataset(include_weight),
        }
    
    @staticmethod
    def numpy_to_ray_dataset(
        X: np.ndarray,
        y: np.ndarray,
        keys: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        """
        直接从 numpy 数组创建 ray.data.Dataset
        
        Args:
            X: 特征矩阵
            y: 标签
            keys: 可选的键数组
            sample_weight: 可选的样本权重
            
        Returns:
            ray.data.Dataset
        """
        try:
            import ray.data
        except ImportError:
            raise ImportError("ray[data] is required. Install with: pip install 'ray[data]'")
        
        data_dict = {"X": X, "y": y}
        if keys is not None:
            data_dict["keys"] = keys
        if sample_weight is not None:
            data_dict["sample_weight"] = sample_weight
        
        return ray.data.from_numpy(data_dict)
