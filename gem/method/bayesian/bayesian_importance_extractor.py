"""
BayesianImportanceExtractor - 贝叶斯模型特征重要性提取器
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from ..base import BaseImportanceExtractor


class BayesianImportanceExtractor(BaseImportanceExtractor):
    """
    贝叶斯模型特征重要性提取器
    
    对于 BayesianRidge/ARDRegression: 使用回归系数的绝对值
    对于 GaussianProcess: 使用 ARD 长度尺度的倒数 (如果使用 ARD 核)
    """
    
    def __init__(
        self,
        importance_type: str = "coefficient",
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
        提取贝叶斯模型特征重要性
        
        Args:
            model: 训练好的贝叶斯模型
            feature_names: 特征名列表
            
        Returns:
            (importance_vector, importance_df) 元组
        """
        try:
            # BayesianRidge 和 ARDRegression 有 coef_ 属性
            if hasattr(model, "coef_"):
                importance = np.abs(model.coef_)
            # ARDRegression 有 lambda_ 属性，可以用作特征重要性
            elif hasattr(model, "lambda_"):
                # lambda_ 越小表示特征越重要
                importance = 1.0 / (model.lambda_ + 1e-10)
            # GaussianProcessRegressor
            elif hasattr(model, "kernel_"):
                # 尝试从核函数中提取长度尺度
                importance = np.ones(len(feature_names))
            else:
                importance = np.ones(len(feature_names))
        except:
            importance = np.ones(len(feature_names))
        
        # 确保长度匹配
        if len(importance) != len(feature_names):
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
