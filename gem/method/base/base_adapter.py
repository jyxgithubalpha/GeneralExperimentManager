"""
DatasetAdapter - Dataset adapter base class

Convert ProcessedViews/SplitView/RayDataBundle to various backend datasets

Process:
1. pl.DataFrame -> numpy (completed in SplitView/RayDataBundle)
2. numpy -> ray.data.Dataset (via RayDataBundle.to_ray_dataset())
3. numpy/ray.data -> backend dataset (e.g., lgb.Dataset)

Supported conversion paths:
- SplitView -> BackendDataset (direct conversion)
- RayDataBundle -> BackendDataset (from numpy conversion)
- RayDataBundle -> ray.data.Dataset -> BackendDataset (distributed)
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
    Dataset adapter base class
    
    Responsible for converting data to specific backend dataset format
    """
    
    @abstractmethod
    def to_dataset(
        self,
        view: "SplitView",
        reference: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Convert SplitView to backend dataset
        
        Args:
            view: SplitView instance
            reference: Reference dataset (for validation/test sets)
            **kwargs: Backend-specific parameters
            
        Returns:
            Backend dataset instance
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
        Convert from RayDataBundle to backend dataset
        
        Args:
            bundle: RayDataBundle instance
            reference: Reference dataset
            **kwargs: Backend-specific parameters
            
        Returns:
            Backend dataset instance
        """
        pass
    
    def to_train_val_test(
        self,
        views: "ProcessedViews",
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """
        Convert ProcessedViews to (train, val, test) tuple
        
        Args:
            views: ProcessedViews instance
            **kwargs: Backend-specific parameters
            
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
        Convert from RayDataViews to (train, val, test) tuple
        
        Args:
            ray_views: RayDataViews instance
            **kwargs: Backend-specific parameters
            
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
        Create datasets dictionary
        
        Args:
            views: ProcessedViews instance
            **kwargs: Backend-specific parameters
            
        Returns:
            {"train": dataset, "val": dataset, "test": dataset}
        """
        dtrain, dval, dtest = self.to_train_val_test(views, **kwargs)
        return {"train": dtrain, "val": dval, "test": dtest}


class RayDataAdapter:
    """
    Ray Data adapter
    
    Responsible for pl.DataFrame/SplitView -> numpy -> ray.data.Dataset conversion
    """
    
    @staticmethod
    def split_view_to_bundle(
        view: "SplitView",
        sample_weight: Optional[np.ndarray] = None,
    ) -> RayDataBundle:
        """
        Convert SplitView to RayDataBundle
        
        Args:
            view: SplitView instance
            sample_weight: Optional sample weights
            
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
        Convert ProcessedViews to RayDataViews
        
        Args:
            views: ProcessedViews instance
            sample_weights: Optional sample weights dictionary {"train": ..., "val": ..., "test": ...}
            
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
        Convert RayDataViews to ray.data.Dataset dictionary
        
        Args:
            ray_views: RayDataViews instance
            include_weight: Whether to include sample weights
            
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
        Create ray.data.Dataset directly from numpy arrays
        
        Args:
            X: Feature matrix
            y: Labels
            keys: Optional key array
            sample_weight: Optional sample weights
            
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
