"""
Data Readers - 数据读取器
包含:
- DataReader: 数据读取器基类
- FeatherReader: Feather 格式读取器"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pyarrow as pa
import polars as pl

from .data_dataclasses import SourceSpec


class DataReader(ABC):
    @abstractmethod
    def read(
        self,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
        filters: Optional[Any] = None,
    ) -> pl.DataFrame:
        pass
    
    def read_sources(
        self,
        source_spec_dict: Dict[str, SourceSpec],
        date_start: int,
        date_end: int,
    ) -> Dict[str, pl.DataFrame]:
        result = {}
        for sourcespec_name, sourcespec in source_spec_dict.items():
            result[sourcespec_name] = self.read(sourcespec, date_start, date_end)
        return result
    
    @staticmethod
    def _standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
        """标准化列名"""
        if "Code" in df.columns and "code" not in df.columns:
            df = df.rename({"Code": "code"})
        
        cols = df.columns
        if "date" in cols and "code" in cols:
            other_cols = [c for c in cols if c not in ("date", "code")]
            df = df.select(["date", "code"] + other_cols)
        
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").cast(pl.Int32))
        if "code" in df.columns:
            df = df.with_columns(pl.col("code").cast(pl.Utf8))
        
        return df


class FeatherReader(DataReader):
    def __init__(self):
        pass
    
    def _process_dataframe(self, df: pl.DataFrame, source_spec: SourceSpec, date_start: int, date_end: int) -> pl.DataFrame:
        if source_spec.name in ["fac"]:
            df = df.with_columns(pl.col("date").cast(pl.Int32))
            df = df.filter((pl.col("date") >= date_start) & (pl.col("date") <= date_end))
        elif source_spec.name in ["ret", "liquidity"]:
            df = df.unpivot(index=['index'], variable_name='code', value_name=f'{source_spec.name}_value')
            df = df.rename({"index": "date"})
            df = df.with_columns(pl.col("date").cast(pl.Int32))
            df = df.filter((pl.col("date") >= date_start) & (pl.col("date") <= date_end))
        elif source_spec.name in ["score", "benchmark", "bench1", "bench2", "bench3", "bench4", "bench5", "bench6"]:
            df = df.unpivot(index=['date'], variable_name='code', value_name=f'{source_spec.name}_value')
            df = df.with_columns(pl.col("date").cast(pl.Int32))
            df = df.filter((pl.col("date") >= date_start) & (pl.col("date") <= date_end))
        return self._standardize_columns(df)

    def read(
        self,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
        filters: Optional[Any] = None,
    ) -> pl.DataFrame:
        feather_reader = pa.ipc.open_file(source_spec.path)
        dfs = []
        
        for i in range(feather_reader.num_record_batches):
            rb = feather_reader.get_record_batch(i)
            df_batch = pl.from_arrow(rb)
            processed_batch = self._process_dataframe(df_batch, source_spec, date_start, date_end)
            
            if processed_batch.height == 0:
                continue
            dfs.append(processed_batch)
        
        if not dfs:
            return pl.DataFrame()
        return pl.concat(dfs)
