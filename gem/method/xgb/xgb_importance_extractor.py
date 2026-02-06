"""
XGBoostImportanceExtractor - XGBoost 特征重要性提取器
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from ..base import BaseImportanceExtractor


class XGBoostImportanceExtractor(BaseImportanceExtractor):
    """
    XGBoost 特征重要性提取器
    
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
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        提取 XGBoost 特征重要性
        
        Args:
            model: 训练好的 XGBoost 模型
            feature_names: 特征名列表
            
        Returns:
            (importance_vector, importance_df) 元组
        """
        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                score_dict = model.get_score(importance_type=self.importance_type)
                importance = np.array([score_dict.get(f, 0) for f in feature_names])
            else:
                importance = np.ones(len(feature_names))
        except:
            importance = np.ones(len(feature_names))
        
        # 归一化
        if self.normalize and np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        # 创建 DataFrame
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        
        return importance, df
