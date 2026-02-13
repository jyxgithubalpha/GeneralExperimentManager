"""
Data preprocessing pipelines for single-source and multi-source tables.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from .utils import remove_quotes_from_list


# =============================================================================
# Single source preprocessors
# =============================================================================


class SingleSourceDataPreprocessor(ABC):
    def fit(self, df: pl.DataFrame) -> "SingleSourceDataPreprocessor":
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class DropDuplicatesPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, key_cols: Optional[List[str]] = None, keep: str = "last"):
        self.key_cols = key_cols or ["date", "code"]
        self.keep = keep

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.unique(subset=self.key_cols, keep=self.keep)


class DropNaNPreprocessor(SingleSourceDataPreprocessor):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(~pl.any_horizontal(pl.all().is_null()))


class FillNaNPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, value: float = 0.0):
        self.value = value

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.fill_null(self.value)


class ColumnFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(
        self,
        keep_cols: Optional[List[str]] = None,
        key_cols: Optional[List[str]] = None,
    ):
        self.keep_cols = keep_cols
        self.key_cols = key_cols or ["date", "code"]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.keep_cols is None:
            return df
        expected = list(dict.fromkeys([*self.key_cols, *self.keep_cols]))
        existing = [col for col in expected if col in df.columns]
        return df.select(existing)


class CodeFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(
        self,
        codes: Optional[List[str]] = None,
        exclude_codes: Optional[List[str]] = None,
    ):
        self.codes = codes
        self.exclude_codes = exclude_codes

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
        exclude_dates: Optional[List[int]] = None,
    ):
        self.exclude_ranges = exclude_ranges or []
        self.exclude_dates = exclude_dates or []

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

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(self.rename_map)


class DTypePreprocessor(SingleSourceDataPreprocessor):
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

    def _parse_dtype(self, dtype_str: Any) -> pl.DataType:
        if isinstance(dtype_str, str):
            dtype_lower = dtype_str.lower()
            if dtype_lower in self.DTYPE_MAP:
                return self.DTYPE_MAP[dtype_lower]
            raise ValueError(f"Unknown dtype: {dtype_str}")
        return dtype_str

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cast_exprs = []
        for col, dtype in self.dtype_map.items():
            if col in df.columns:
                cast_exprs.append(pl.col(col).cast(self._parse_dtype(dtype)))
        if cast_exprs:
            return df.with_columns(cast_exprs)
        return df


class SingleSourceDataPreprocessorPipeline:
    def __init__(self, steps: Dict[str, SingleSourceDataPreprocessor]):
        self.steps = steps

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df
        for step in self.steps.values():
            result = step.fit(result).transform(result)
        return result


# =============================================================================
# Multi source preprocessors
# =============================================================================


class MultiSourceDataPreprocessor(ABC):
    def fit(self, source_dict: Dict[str, pl.DataFrame]) -> "MultiSourceDataPreprocessor":
        return self

    def transform(self, source_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        return source_dict


class AlignPreprocessor(MultiSourceDataPreprocessor):
    def __init__(
        self,
        key_cols: Optional[List[str]] = None,
        align_key_train_source_list: Optional[List[str]] = None,
        align_key_eval_source_list: Optional[List[str]] = None,
    ):
        self.key_cols = remove_quotes_from_list(key_cols) or ["date", "code"]
        self._align_key_train_source_list = remove_quotes_from_list(
            align_key_train_source_list
        )
        self._align_key_eval_source_list = remove_quotes_from_list(
            align_key_eval_source_list
        )
        self._align_key_train: Optional[pl.DataFrame] = None

    def _collect_inner_keys(self, source_dict: Dict[str, pl.DataFrame], source_names: List[str]) -> Optional[pl.DataFrame]:
        keys = None
        for source_name in source_names:
            if source_name not in source_dict:
                continue
            keys_df = source_dict[source_name].select(self.key_cols).unique()
            keys = keys_df if keys is None else keys.join(keys_df, on=self.key_cols, how="inner")
        return keys

    def fit(self, source_dict: Dict[str, pl.DataFrame]) -> "AlignPreprocessor":
        keys_train = self._collect_inner_keys(source_dict, self._align_key_train_source_list)
        if keys_train is None:
            self._align_key_train = None
        else:
            self._align_key_train = keys_train.sort(self.key_cols)
        return self

    def transform(self, source_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        if self._align_key_train is None:
            return source_dict

        result: Dict[str, pl.DataFrame] = {}
        for source_name, frame in source_dict.items():
            if source_name in self._align_key_train_source_list:
                result[source_name] = frame.join(self._align_key_train, on=self.key_cols, how="inner")
            elif source_name in self._align_key_eval_source_list:
                result[source_name] = self._align_key_train.join(frame, on=self.key_cols, how="left")
            else:
                result[source_name] = frame
        return result


class MultiSourceDataPreprocessorPipeline:
    def __init__(self, steps: Dict[str, MultiSourceDataPreprocessor]):
        self.steps = steps

    def fit_transform(self, source_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        result = source_dict
        for step in self.steps.values():
            result = step.fit(result).transform(result)
        return result
