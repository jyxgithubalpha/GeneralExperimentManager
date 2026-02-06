"""
Data Preprocessors - 数据预处理器

包含:
- SingleSourceDataPreprocessor: 单源预处理器基类
- 各种单源预处理器实现
- MultiSourceDataPreprocessor: 多源预处理器基类
- AlignPreprocessor: 对齐预处理器
- Pipeline 类
"""
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from .utils import remove_quotes_from_list


# =============================================================================
# Single Source Preprocessors
# =============================================================================

class SingleSourceDataPreprocessor(ABC):
    def fit(self, df: pd.DataFrame) -> 'SingleSourceDataPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class DropDuplicatesPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, key_cols: List[str] = None, keep: str = "last"):
        self.key_cols = key_cols or ["date", "code"]
        self.keep = keep
    
    def fit(self, df: pd.DataFrame) -> 'DropDuplicatesPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=self.key_cols, keep=self.keep)


class DropNaNPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, subset: Optional[List[str]] = None, how: str = "any"):
        self.subset = subset
        self.how = how
    
    def fit(self, df: pd.DataFrame) -> 'DropNaNPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=self.subset, how=self.how)


class FillNaNPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, value: float = 0.0, method: Optional[str] = None):
        self.value = value
        self.method = method
    
    def fit(self, df: pd.DataFrame) -> 'FillNaNPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method:
            return df.fillna(method=self.method)
        return df.fillna(self.value)


class ColumnFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, keep_cols: Optional[List[str]] = None, key_cols: List[str] = None):
        self.keep_cols = keep_cols
        self.key_cols = key_cols or ["date", "code"]
    
    def fit(self, df: pd.DataFrame) -> 'ColumnFilterPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.keep_cols is None:
            return df
        expected = list(dict.fromkeys([*self.key_cols, *self.keep_cols]))
        existing = [c for c in expected if c in df.columns]
        return df[existing].copy()


class CodeFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, codes: Optional[List[str]] = None, exclude_codes: Optional[List[str]] = None):
        self.codes = codes
        self.exclude_codes = exclude_codes
    
    def fit(self, df: pd.DataFrame) -> 'CodeFilterPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.codes is not None:
            df = df[df["code"].isin(self.codes)]
        if self.exclude_codes is not None:
            df = df[~df["code"].isin(self.exclude_codes)]
        return df.copy()


class DateFilterPreprocessor(SingleSourceDataPreprocessor):
    def __init__(
        self, 
        exclude_ranges: Optional[List[Tuple[int, int]]] = None,
        exclude_dates: Optional[List[int]] = None
    ):
        self.exclude_ranges = exclude_ranges or []
        self.exclude_dates = exclude_dates or []
    
    def fit(self, df: pd.DataFrame) -> 'DateFilterPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        
        for start, end in self.exclude_ranges:
            mask &= ~((df["date"] >= start) & (df["date"] <= end))
        
        if self.exclude_dates:
            mask &= ~df["date"].isin(self.exclude_dates)
        
        return df[mask].copy()


class RenameColumnsPreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, rename_map: Dict[str, str]):
        self.rename_map = rename_map
    
    def fit(self, df: pd.DataFrame) -> 'RenameColumnsPreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.rename_map)


class DTypePreprocessor(SingleSourceDataPreprocessor):
    def __init__(self, dtype_map: Dict[str, Any]):
        self.dtype_map = dtype_map
    
    def fit(self, df: pd.DataFrame) -> 'DTypePreprocessor':
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, dtype in self.dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df


class SingleSourceDataPreprocessorPipeline:
    def __init__(self, steps: Dict[str, SingleSourceDataPreprocessor]):
        self.steps = steps
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        for step_name, step in self.steps.items():
            result = step.fit(result).transform(result)
        return result


# =============================================================================
# Multi Source Preprocessors
# =============================================================================

class MultiSourceDataPreprocessor(ABC):
    def fit(self, source_dict: Dict[str, pd.DataFrame]) -> 'MultiSourceDataPreprocessor':
        return self
    
    def transform(self, source_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return source_dict


class AlignPreprocessor(MultiSourceDataPreprocessor):
    def __init__(self, key_cols: list[str] = None, align_key_train_source_list: list[str] = None, align_key_eval_source_list: list[str] = None):
        self.key_cols = remove_quotes_from_list(key_cols)
        self._align_key_train_source_list = remove_quotes_from_list(align_key_train_source_list)
        self._align_key_train: pd.DataFrame | None = None
        self._align_key_eval_source_list = remove_quotes_from_list(align_key_eval_source_list)
        self._align_key_eval: pd.DataFrame | None = None

    def fit(self, source_dict: dict[str, pd.DataFrame]) -> "AlignPreprocessor":
        keys_train = None
        keys_eval = None
        for df_name, df in source_dict.items():
            if df_name in self._align_key_train_source_list:
                idx = pd.MultiIndex.from_frame(df[self.key_cols])
                keys_train = idx if keys_train is None else keys_train.intersection(idx)
            if df_name in self._align_key_eval_source_list:
                idx = pd.MultiIndex.from_frame(df[self.key_cols])
                keys_eval = idx if keys_eval is None else keys_eval.intersection(idx)
        
        self._align_key_train = keys_train.to_frame(index=False).sort_values(by=self.key_cols).reset_index(drop=True) if keys_train is not None else pd.DataFrame(columns=self.key_cols)
        self._align_key_eval = keys_eval.to_frame(index=False).sort_values(by=self.key_cols).reset_index(drop=True) if keys_eval is not None else pd.DataFrame(columns=self.key_cols)
        
        return self

    def transform(self, source_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        result = {}
        for df_name, df in source_dict.items():
            if df_name in self._align_key_train_source_list:
                result[df_name] = df.merge(self._align_key_train, on=self.key_cols, how="inner")
            elif df_name in self._align_key_eval_source_list:
                result[df_name] = self._align_key_train.merge(df, on=self.key_cols, how="left")
            else:
                result[df_name] = df
        
        return result


class MultiSourceDataPreprocessorPipeline:
    def __init__(self, steps: Dict[str, MultiSourceDataPreprocessor]):
        self.steps = steps
    
    def fit_transform(self, source_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        result = source_dict
        for step_name, step in self.steps.items():
            result = step.fit(result).transform(result)
        return result
