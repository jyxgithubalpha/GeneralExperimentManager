"""
Data Assemblers - 数据组装器
包含:
- GlobalDataAssembler: 全局数据组装器基类
- FeatureAssembler: 特征组装器"""



from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import polars as pl

from .data_dataclasses import DatasetSpec, GlobalStore
from .utils import remove_quotes_from_list


class GlobalDataAssembler(ABC):
    """全局数据组装器基类"""
    
    @abstractmethod
    def assemble(self, source_dict: Dict[str, pl.DataFrame]) -> GlobalStore:
        """组装数据源为 GlobalStore"""
        pass


class FeatureAssembler(GlobalDataAssembler):
    """特征组装器- 将多个数据源组装为 GlobalStore"""
    
    def __init__(self, dataset_spec: DatasetSpec):
        self.X_source_list = remove_quotes_from_list(dataset_spec.X_source_list)
        self.y_source_list = remove_quotes_from_list(dataset_spec.y_source_list)
        self.extra_source_list = remove_quotes_from_list(dataset_spec.extra_source_list)
        self.key_cols = remove_quotes_from_list(dataset_spec.key_cols)
        self.group_col = remove_quotes_from_list(dataset_spec.group_col)
    
    def _prefix_non_key_cols(self, df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        """为非主键列添加前缀"""
        rename_map = {
            col: f"{prefix}__{col}" 
            for col in df.columns 
            if col not in self.key_cols
        }
        return df.rename(rename_map)
    
    def assemble(self, source_dict: Dict[str, pl.DataFrame]) -> GlobalStore:
        """组装数据源为 GlobalStore"""
        # 组装 X
        X_dfs = [self._prefix_non_key_cols(source_dict[name], name) for name in self.X_source_list]
        X_df = X_dfs[0]
        for df in X_dfs[1:]:
            X_df = X_df.join(df, on=self.key_cols, how="outer_coalesce")
        
        # 组装 y
        y_dfs = [self._prefix_non_key_cols(source_dict[name], name) for name in self.y_source_list]
        y_df = y_dfs[0]
        for df in y_dfs[1:]:
            y_df = y_df.join(df, on=self.key_cols, how="outer_coalesce")
        
        # 组装 extra
        extra_df = None
        if self.extra_source_list:
            extra_dfs = [self._prefix_non_key_cols(source_dict[name], name) for name in self.extra_source_list]
            extra_df = extra_dfs[0]
            for df in extra_dfs[1:]:
                extra_df = extra_df.join(df, on=self.key_cols, how="outer_coalesce")
        
        # 对齐所有数据 (以 X_df 的 keys 为基准)
        keys = X_df.select(self.key_cols)
        y_df = keys.join(y_df, on=self.key_cols, how="left")
        
        if extra_df is not None:
            extra_df = keys.join(extra_df, on=self.key_cols, how="left")
        
        # 提取特征名和标签名
        feature_names = [c for c in X_df.columns if c not in self.key_cols]
        label_names = [c for c in y_df.columns if c not in self.key_cols]
        
        # 转换为 numpy
        X_full = X_df.select(feature_names).to_numpy().astype(np.float32)
        y_full = y_df.select(label_names).to_numpy().astype(np.float32)
        
        return GlobalStore(
            keys=keys,
            X_full=X_full,
            y_full=y_full,
            feature_name_list=feature_names,
            label_name_list=label_names,
            extra=extra_df,
        )
