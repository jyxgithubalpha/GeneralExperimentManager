"""
BaseImportanceExtractor - 特征重要性提取器基类
"""
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import polars as pl


class BaseImportanceExtractor(ABC):
    """特征重要性提取器基类"""
    
    @abstractmethod
    def extract(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, pl.DataFrame]:
        """提取特征重要性"""
        pass
