"""
DatasetAdapter - 数据集适配器基类

将 ProcessedViews/SplitView 转换为各后端 dataset
"""



from abc import ABC, abstractmethod
from typing import Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews, SplitView


class DatasetAdapter(ABC):
    """数据集适配器基类"""
    
    @abstractmethod
    def to_dataset(self, view: "SplitView", **kwargs) -> Any:
        """将 SplitView 转换为后端 dataset"""
        pass
    
    @abstractmethod
    def to_train_val_test(
        self,
        views: "ProcessedViews",
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """将 ProcessedViews 转换为 (train, val, test) 三元组"""
        pass
