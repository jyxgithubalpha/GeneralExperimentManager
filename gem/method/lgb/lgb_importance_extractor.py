"""
LightGBMImportanceExtractor - LightGBM 特征重要性提取器
"""


from typing import Any, List, Tuple

import numpy as np
import polars as pl

from ..base import BaseImportanceExtractor


class LightGBMImportanceExtractor(BaseImportanceExtractor):
    """
    LightGBM 特征重要性提取器
    
    统一输出 per-feature vector，与当前 feature_names 对齐
    """
    
    def __init__(
        self,
        importance_type: str = "gain",
        normalize: bool = True,
    ):
        self.importance_type = importance_type
        self.normalize = normalize
    
    def extract(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, pl.DataFrame]:
        """
        提取 LightGBM 特征重要性
        
        Args:
            model: 训练好的 LightGBM 模型
            feature_names: 特征名列表
            
        Returns:
            (importance_vector, importance_df) 元组
        """
        try:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                importance = model.feature_importance(importance_type=self.importance_type)
            else:
                importance = np.ones(len(feature_names))
        except:
            importance = np.ones(len(feature_names))
        
        # 归一化
        if self.normalize and np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        # 创建 DataFrame
        df = pl.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort("importance", descending=True)
        
        return importance, df
