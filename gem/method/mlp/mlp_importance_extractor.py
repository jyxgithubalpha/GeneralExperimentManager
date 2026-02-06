"""
MLPImportanceExtractor - MLP 特征重要性提取器
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from ..base import BaseImportanceExtractor


class MLPImportanceExtractor(BaseImportanceExtractor):
    """
    MLP 特征重要性提取器
    
    使用输入层权重的绝对值作为特征重要性的近似
    """
    
    def __init__(
        self,
        importance_type: str = "weight",
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
        提取 MLP 特征重要性
        
        Args:
            model: 训练好的 MLP 模型
            feature_names: 特征名列表
            
        Returns:
            (importance_vector, importance_df) 元组
        """
        try:
            import torch
            # 获取第一层权重
            first_layer = None
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    first_layer = module
                    break
            
            if first_layer is not None:
                weights = first_layer.weight.data.cpu().numpy()
                # 使用权重的绝对值平均作为重要性
                importance = np.abs(weights).mean(axis=0)
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
