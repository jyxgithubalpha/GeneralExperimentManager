"""
Data Readers - 数据读取器

包含:
- DataReader: 数据读取器基类
- FeatherReader: Feather 格式读取器
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from .data_dataclasses import SourceSpec


class DataReader(ABC):
    @abstractmethod
    def read(
        self,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
        filters: Optional[Any] = None,
    ) -> pd.DataFrame:
        pass
    
    def read_sources(
        self,
        source_spec_dict: Dict[str, SourceSpec],
        date_start: int,
        date_end: int,
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for sourcespec_name, sourcespec in source_spec_dict.items():
            result[sourcespec_name] = self.read(sourcespec, date_start, date_end)
        return result
    
    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        if "Code" in df.columns and "code" not in df.columns:
            df = df.rename(columns={"Code": "code"})
        
        cols = df.columns.tolist()
        if "date" in cols and "code" in cols:
            other_cols = [c for c in cols if c not in ("date", "code")]
            df = df[["date", "code"] + other_cols]
        
        if "date" in df.columns:
            df["date"] = df["date"].astype(np.int32)
        if "code" in df.columns:
            df["code"] = df["code"].astype(str)
        
        return df


class FeatherReader(DataReader):
    def __init__(self, is_on_server: bool = False):
        self.is_on_server = is_on_server
    
    def _process_dataframe(self, df: pd.DataFrame, source_spec: SourceSpec, date_start: int, date_end: int) -> pd.DataFrame:
        if source_spec.name in ["fac"]:
            df["date"] = df["date"].astype(np.int32)
            df = df[(df["date"] >= date_start) & (df["date"] <= date_end)]
        elif source_spec.name in ["ret", "liquidity"]:
            df = df.melt(id_vars=['index'], var_name='code', value_name=f'{source_spec.name}_value')
            df = df.rename(columns={"index": "date"})
            df["date"] = df["date"].astype(np.int32)
            df = df[(df["date"] >= date_start) & (df["date"] <= date_end)]
        elif source_spec.name in ["score", "benchmark", "bench1", "bench2", "bench3", "bench4", "bench5", "bench6"]:
            df = df.melt(id_vars=['date'], var_name='code', value_name=f'{source_spec.name}_value')
            df["date"] = df["date"].astype(np.int32)
            df = df[(df["date"] >= date_start) & (df["date"] <= date_end)]
        return df

    def read(
        self,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
    ) -> pd.DataFrame:
        if self.is_on_server:
            df = pd.read_feather(source_spec.path)
            return self._process_dataframe(df, source_spec, date_start, date_end)
        
        feather_reader = pa.ipc.open_file(source_spec.path)
        batches = []
        
        for i in range(feather_reader.num_record_batches):
            rb = feather_reader.get_record_batch(i)
            df_batch = rb.to_pandas()
            processed_batch = self._process_dataframe(df_batch, source_spec, date_start, date_end)
            
            if len(processed_batch) > 0:
                batches.append(processed_batch)
        
        if batches:
            df = pd.concat(batches, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        return self._standardize_columns(df)
