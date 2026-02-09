"""
LightGBMAdapter - LightGBM 数据适配器
将 SplitView 转换为 lgb.Dataset
"""



from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..base.base_adapter import DatasetAdapter

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews, SplitView


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
        view: "SplitView",
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
        views: "ProcessedViews",
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """转换为 (train_dataset, val_dataset, test_dataset)"""
        dtrain = self.to_dataset(views.train, **kwargs)
        dval = self.to_dataset(views.val, reference=dtrain, **kwargs)
        dtest = self.to_dataset(views.test, reference=dtrain, **kwargs)
        
        return dtrain, dval, dtest
    
    def create_datasets_dict(
        self,
        views: "ProcessedViews",
        **kwargs
    ) -> Dict[str, Any]:
        """创建 datasets 字典，用于 feval 等需要映射的场景"""
        dtrain, dval, dtest = self.to_train_val_test(views, **kwargs)
        return {
            "train": dtrain,
            "val": dval,
            "test": dtest,
        }
