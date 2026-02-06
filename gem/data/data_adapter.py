"""
Adapter - 后端适配器

将 ProcessedViews/SplitView 转换为各后端 dataset:
- LightGBM: lgb.Dataset
- PyTorch: torch.utils.data.Dataset / DataLoader
- sklearn: numpy arrays

Adapter 仅做格式转换，不做拟合。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_dataclasses import ProcessedViews, SplitView


# =============================================================================
# 1. 基类
# =============================================================================

class DatasetAdapter(ABC):
    """数据集适配器基类"""
    
    @abstractmethod
    def to_dataset(self, view: SplitView, **kwargs) -> Any:
        """将 SplitView 转换为后端 dataset"""
        pass
    
    @abstractmethod
    def to_train_val_test(
        self,
        views: ProcessedViews,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """将 ProcessedViews 转换为 (train, val, test) 三元组"""
        pass


# =============================================================================
# 2. LightGBM Adapter
# =============================================================================

class LightGBMAdapter(DatasetAdapter):
    """
    LightGBM 数据集适配器
    
    将 SplitView 转换为 lgb.Dataset
    """
    
    def __init__(
        self,
        feature_name: str = "auto",
        categorical_feature: str = "auto",
        free_raw_data: bool = True,
    ):
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.free_raw_data = free_raw_data
    
    def to_dataset(
        self,
        view: SplitView,
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        转换为 lgb.Dataset
        
        Args:
            view: SplitView 实例
            reference: 参考 dataset (用于验证集)
            
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
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """转换为 (train_dataset, val_dataset, test_dataset)"""
        dtrain = self.to_dataset(views.train, **kwargs)
        dval = self.to_dataset(views.val, reference=dtrain, **kwargs)
        dtest = self.to_dataset(views.test, reference=dtrain, **kwargs)
        
        return dtrain, dval, dtest
    
    def create_datasets_dict(
        self,
        views: ProcessedViews,
        **kwargs
    ) -> Dict[str, Any]:
        """创建 datasets 字典，用于 feval 等需要映射的场景"""
        dtrain, dval, dtest = self.to_train_val_test(views, **kwargs)
        return {
            "train": dtrain,
            "val": dval,
            "test": dtest,
        }


# =============================================================================
# 3. NumPy Adapter (for sklearn)
# =============================================================================

class NumpyAdapter(DatasetAdapter):
    """
    NumPy 数据适配器
    
    将 SplitView 转换为 (X, y) numpy 数组元组
    """
    
    def __init__(self, dtype: np.dtype = np.float32):
        self.dtype = dtype
    
    def to_dataset(
        self,
        view: SplitView,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """转换为 (X, y) 元组"""
        X = view.X.astype(self.dtype)
        y = view.y.ravel().astype(self.dtype) if view.y.ndim > 1 else view.y.astype(self.dtype)
        return X, y
    
    def to_train_val_test(
        self,
        views: ProcessedViews,
        **kwargs
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """转换为 ((X_train, y_train), (X_val, y_val), (X_test, y_test))"""
        return (
            self.to_dataset(views.train),
            self.to_dataset(views.val),
            self.to_dataset(views.test),
        )
    
    def to_dict(self, views: ProcessedViews, **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
        """转换为字典格式"""
        train, val, test = self.to_train_val_test(views, **kwargs)
        return {
            "train": {"X": train[0], "y": train[1]},
            "val": {"X": val[0], "y": val[1]},
            "test": {"X": test[0], "y": test[1]},
        }


# =============================================================================
# 4. PyTorch Adapter
# =============================================================================

class PyTorchDataset:
    """PyTorch Dataset wrapper"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        extra: Optional[pd.DataFrame] = None,
    ):
        self.X = X
        self.y = y
        self.extra = extra
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        try:
            import torch
            X = torch.from_numpy(self.X[idx]).float()
            y = torch.from_numpy(np.array([self.y[idx]])).float()
            return X, y
        except ImportError:
            return self.X[idx], self.y[idx]


class PyTorchAdapter(DatasetAdapter):
    """
    PyTorch 数据适配器
    
    将 SplitView 转换为 PyTorch Dataset/DataLoader
    """
    
    def __init__(
        self,
        batch_size: int = 1024,
        shuffle_train: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def to_dataset(
        self,
        view: SplitView,
        **kwargs
    ) -> PyTorchDataset:
        """转换为 PyTorchDataset"""
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        return PyTorchDataset(
            X=view.X.astype(np.float32),
            y=y.astype(np.float32),
            extra=view.extra,
        )
    
    def to_dataloader(
        self,
        view: SplitView,
        shuffle: bool = False,
        **kwargs
    ) -> Any:
        """转换为 DataLoader"""
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("torch is required for PyTorchAdapter.to_dataloader")
        
        dataset = self.to_dataset(view)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **kwargs
        )
    
    def to_train_val_test(
        self,
        views: ProcessedViews,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """转换为 (train_loader, val_loader, test_loader)"""
        train_loader = self.to_dataloader(views.train, shuffle=self.shuffle_train)
        val_loader = self.to_dataloader(views.val, shuffle=False)
        test_loader = self.to_dataloader(views.test, shuffle=False)
        
        return train_loader, val_loader, test_loader


# =============================================================================
# 5. XGBoost Adapter
# =============================================================================

class XGBoostAdapter(DatasetAdapter):
    """
    XGBoost 数据适配器
    
    将 SplitView 转换为 xgb.DMatrix
    """
    
    def __init__(self, enable_categorical: bool = False):
        self.enable_categorical = enable_categorical
    
    def to_dataset(
        self,
        view: SplitView,
        **kwargs
    ) -> Any:
        """转换为 xgb.DMatrix"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for XGBoostAdapter")
        
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        
        return xgb.DMatrix(
            data=view.X,
            label=y,
            feature_names=view.feature_names,
            enable_categorical=self.enable_categorical,
            **kwargs
        )
    
    def to_train_val_test(
        self,
        views: ProcessedViews,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """转换为 (dtrain, dval, dtest)"""
        return (
            self.to_dataset(views.train, **kwargs),
            self.to_dataset(views.val, **kwargs),
            self.to_dataset(views.test, **kwargs),
        )


# =============================================================================
# 6. CatBoost Adapter
# =============================================================================

class CatBoostAdapter(DatasetAdapter):
    """
    CatBoost 数据适配器
    
    将 SplitView 转换为 catboost.Pool
    """
    
    def __init__(self, cat_features: Optional[List[int]] = None):
        self.cat_features = cat_features
    
    def to_dataset(
        self,
        view: SplitView,
        **kwargs
    ) -> Any:
        """转换为 catboost.Pool"""
        try:
            from catboost import Pool
        except ImportError:
            raise ImportError("catboost is required for CatBoostAdapter")
        
        y = view.y.ravel() if view.y.ndim > 1 else view.y
        
        return Pool(
            data=view.X,
            label=y,
            feature_names=view.feature_names,
            cat_features=self.cat_features,
            **kwargs
        )
    
    def to_train_val_test(
        self,
        views: ProcessedViews,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """转换为 (train_pool, val_pool, test_pool)"""
        return (
            self.to_dataset(views.train, **kwargs),
            self.to_dataset(views.val, **kwargs),
            self.to_dataset(views.test, **kwargs),
        )


# =============================================================================
# 7. 适配器工厂
# =============================================================================

class AdapterFactory:
    """适配器工厂"""
    
    _registry: Dict[str, type] = {
        "lightgbm": LightGBMAdapter,
        "lgb": LightGBMAdapter,
        "numpy": NumpyAdapter,
        "sklearn": NumpyAdapter,
        "pytorch": PyTorchAdapter,
        "torch": PyTorchAdapter,
        "xgboost": XGBoostAdapter,
        "xgb": XGBoostAdapter,
        "catboost": CatBoostAdapter,
        "cb": CatBoostAdapter,
    }
    
    @classmethod
    def create(cls, backend: str, **kwargs) -> DatasetAdapter:
        """
        创建适配器
        
        Args:
            backend: 后端名称 (lightgbm, numpy, pytorch, xgboost, catboost)
            **kwargs: 适配器参数
            
        Returns:
            DatasetAdapter 实例
        """
        backend_lower = backend.lower()
        if backend_lower not in cls._registry:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(cls._registry.keys())}")
        
        return cls._registry[backend_lower](**kwargs)
    
    @classmethod
    def register(cls, name: str, adapter_class: type) -> None:
        """注册新适配器"""
        cls._registry[name.lower()] = adapter_class


# =============================================================================
# 8. 便捷函数
# =============================================================================

def to_lgb_datasets(
    views: ProcessedViews,
    **kwargs
) -> Tuple[Any, Any, Any]:
    """快速转换为 LightGBM datasets"""
    adapter = LightGBMAdapter(**kwargs)
    return adapter.to_train_val_test(views)


def to_numpy(
    views: ProcessedViews,
    dtype: np.dtype = np.float32,
) -> Dict[str, Dict[str, np.ndarray]]:
    """快速转换为 numpy 字典"""
    adapter = NumpyAdapter(dtype=dtype)
    return adapter.to_dict(views)


def to_pytorch_loaders(
    views: ProcessedViews,
    batch_size: int = 1024,
    **kwargs
) -> Tuple[Any, Any, Any]:
    """快速转换为 PyTorch DataLoaders"""
    adapter = PyTorchAdapter(batch_size=batch_size, **kwargs)
    return adapter.to_train_val_test(views)
