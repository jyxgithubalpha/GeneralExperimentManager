"""
Data Preprocessors - 数据预处理器

包含:
- SingleSourceDataPreprocessor: 单源预处理器基类
- 各种单源预处理器实现
- MultiSourceDataPreprocessor: 多源预处理器基类
- AlignPreprocessor: 对齐预处理器
- Pipeline: 预处理器流水线"""

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from .utils import remove_quotes_from_list


# =============================================================================
# Single Source Preprocessors
# =============================================================================

class SingleSourceDataPreprocessor(ABC):
    def fit(self, df: pl.DataFrame) -> 'SingleSourceDataPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class DropDuplicatesPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, key_cols: List[str] = None, keep: str = "last"):
        self.key_cols = key_cols or ["date", "code"]
        self.keep = keep
    
    def fit(self, df: pl.DataFrame) -> 'DropDuplicatesPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.unique(subset=self.key_cols, keep=self.keep)


class DropNaNPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self):
        pass    
    
    def fit(self, df: pl.DataFrame) -> 'DropNaNPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(~pl.any_horizontal(pl.all().is_null()))


class FillNaNPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def fit(self, df: pl.DataFrame) -> 'FillNaNPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.fill_null(self.value)


class ColumnFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, keep_cols: Optional[List[str]] = None, key_cols: List[str] = None):
        self.keep_cols = keep_cols
        self.key_cols = key_cols or ["date", "code"]
    
    def fit(self, df: pl.DataFrame) -> 'ColumnFilterPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.keep_cols is None:
            return df
        expected = list(dict.fromkeys([*self.key_cols, *self.keep_cols]))
        existing = [c for c in expected if c in df.columns]
        return df.select(existing)


class CodeFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, codes: Optional[List[str]] = None, exclude_codes: Optional[List[str]] = None):
        self.codes = codes
        self.exclude_codes = exclude_codes
    
    def fit(self, df: pl.DataFrame) -> 'CodeFilterPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.codes is not None:
            df = df.filter(pl.col("code").is_in(self.codes))
        if self.exclude_codes is not None:
            df = df.filter(~pl.col("code").is_in(self.exclude_codes))
        return df


class DateFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(
        self, 
        exclude_ranges: Optional[List[Tuple[int, int]]] = None,
        exclude_dates: Optional[List[int]] = None
    ):
        self.exclude_ranges = exclude_ranges or []
        self.exclude_dates = exclude_dates or []
    
    def fit(self, df: pl.DataFrame) -> 'DateFilterPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        mask = pl.lit(True)
        
        for start, end in self.exclude_ranges:
            mask = mask & ~((pl.col("date") >= start) & (pl.col("date") <= end))
        
        if self.exclude_dates:
            mask = mask & ~pl.col("date").is_in(self.exclude_dates)
        
        return df.filter(mask)


class RenameColumnsPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, rename_map: Dict[str, str]):
        self.rename_map = rename_map
    
    def fit(self, df: pl.DataFrame) -> 'RenameColumnsPreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(self.rename_map)


class DTypePreprocessor(SingleSourceDataPreprocessor):
    # String to Polars dtype mapping
    DTYPE_MAP = {
        "int8": pl.Int8,
        "int16": pl.Int16,
        "int32": pl.Int32,
        "int64": pl.Int64,
        "uint8": pl.UInt8,
        "uint16": pl.UInt16,
        "uint32": pl.UInt32,
        "uint64": pl.UInt64,
        "float32": pl.Float32,
        "float64": pl.Float64,
        "str": pl.Utf8,
        "string": pl.Utf8,
        "utf8": pl.Utf8,
        "bool": pl.Boolean,
        "date": pl.Date,
        "datetime": pl.Datetime,
    }
    
    def __init__(self, dtype_map: Dict[str, Any]):
        self.dtype_map = dtype_map
    
    def _parse_dtype(self, dtype_str: str) -> pl.DataType:
        """Convert string dtype to Polars dtype"""
        if isinstance(dtype_str, str):
            dtype_lower = dtype_str.lower()
            if dtype_lower in self.DTYPE_MAP:
                return self.DTYPE_MAP[dtype_lower]
            raise ValueError(f"Unknown dtype: {dtype_str}")
        return dtype_str  # Already a Polars dtype
    
    def fit(self, df: pl.DataFrame) -> 'DTypePreprocessor':
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cast_exprs = []
        for col, dtype in self.dtype_map.items():
            if col in df.columns:
                pl_dtype = self._parse_dtype(dtype)
                cast_exprs.append(pl.col(col).cast(pl_dtype))
        if cast_exprs:
            return df.with_columns(cast_exprs)
        return df


class SingleSourceDataPreprocessorPipeline:
    def __init__(self, steps: Dict[str, SingleSourceDataPreprocessor]):
        self.steps = steps
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df
        for step_name, step in self.steps.items():
            result = step.fit(result).transform(result)
        return result


# =============================================================================
# Multi Source Preprocessors
# =============================================================================

class MultiSourceDataPreprocessor(ABC):
    def fit(self, source_dict: Dict[str, pl.DataFrame]) -> 'MultiSourceDataPreprocessor':
        return self
    
    def transform(self, source_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        return source_dict


class AlignPreprocessor(MultiSourceDataPreprocessor):
    def __init__(self, key_cols: list[str] = None, align_key_train_source_list: list[str] = None, align_key_eval_source_list: list[str] = None):
        self.key_cols = remove_quotes_from_list(key_cols)
        self._align_key_train_source_list = remove_quotes_from_list(align_key_train_source_list)
        self._align_key_train: pl.DataFrame | None = None
        self._align_key_eval_source_list = remove_quotes_from_list(align_key_eval_source_list)
        self._align_key_eval: pl.DataFrame | None = None

    def fit(self, source_dict: dict[str, pl.DataFrame]) -> "AlignPreprocessor":
        keys_train = None
        keys_eval = None
        for df_name, df in source_dict.items():
            if df_name in self._align_key_train_source_list:
                keys_df = df.select(self.key_cols).unique()
                keys_train = keys_df if keys_train is None else keys_train.join(keys_df, on=self.key_cols, how="inner")
            if df_name in self._align_key_eval_source_list:
                keys_df = df.select(self.key_cols).unique()
                keys_eval = keys_df if keys_eval is None else keys_eval.join(keys_df, on=self.key_cols, how="inner")
        
        self._align_key_train = keys_train.sort(self.key_cols) if keys_train is not None else pl.DataFrame(schema={col: pl.Utf8 for col in self.key_cols})
        self._align_key_eval = keys_eval.sort(self.key_cols) if keys_eval is not None else pl.DataFrame(schema={col: pl.Utf8 for col in self.key_cols})
        
        return self

    def transform(self, source_dict: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
        result = {}
        for df_name, df in source_dict.items():
            if df_name in self._align_key_train_source_list:
                result[df_name] = df.join(self._align_key_train, on=self.key_cols, how="inner")
            elif df_name in self._align_key_eval_source_list:
                result[df_name] = self._align_key_train.join(df, on=self.key_cols, how="left")
            else:
                result[df_name] = df
        
        return result


class MultiSourceDataPreprocessorPipeline:
    def __init__(self, steps: Dict[str, MultiSourceDataPreprocessor]):
        self.steps = steps
    
    def fit_transform(self, source_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        result = source_dict
        for step_name, step in self.steps.items():
            result = step.fit(result).transform(result)
        return result
