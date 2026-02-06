"""
核心数据结构定义 - 剩余的通用数据类

包含:
- ProcessedViews: DataProcessor 处理后的视图
- SplitResult: Split 执行结果
- TransformState: 变换状态
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitSpec:
    split_id: int
    train_date_list: List[int]
    val_date_list: List[int]
    test_date_list: List[int]
    anchor_time: Optional[str] = None  # e.g., "2023Q1"
    
    def get_all_dates_range(self) -> Tuple[int, int]:
        all_dates = self.train_date_list + self.val_date_list + self.test_date_list
        return min(all_dates), max(all_dates)


@dataclass
class SplitGeneratorOutput:
    splitspec_list: List[SplitSpec]
    date_start: int
    date_end: int


@dataclass
class SourceSpec:
    name: str
    format: str = "feather"
    path: str = ""


@dataclass
class DatasetSpec:
    X_source_list: List[str] = field(default_factory=list)
    y_source_list: List[str] = field(default_factory=list)
    extra_source_list: List[str] = field(default_factory=list)
    key_cols: List[str] = field(default_factory=list)
    group_col: str = None


@dataclass
class GlobalStore:
    keys: pd.DataFrame
    X_full: np.ndarray
    y_full: np.ndarray
    
    feature_name_list: List[str]
    label_name_list: List[str]
    date_col: str = "date"
    code_col: str = "code"
    
    extra: Optional[pd.DataFrame] = None
    
    _date_to_indices: Optional[Dict[int, np.ndarray]] = field(default=None, init=False)
    
    def __post_init__(self):
        """数据一致性检查和索引预计算"""
        n_samples = len(self.keys)
        assert self.X_full.shape[0] == n_samples, "X_full 行数不匹配"
        assert self.y_full.shape[0] == n_samples, "y_full 行数不匹配"
        assert self.X_full.shape[1] == len(self.feature_name_list), "特征名数量不匹配"
        
        assert self.date_col in self.keys.columns, f"缺少日期列: {self.date_col}"
        assert self.code_col in self.keys.columns, f"缺少代码列: {self.code_col}"
        
        self._build_date_index()
    
    def _build_date_index(self):
        """预计算日期到行索引的映射"""
        self._date_to_indices = {}
        for date in self.keys[self.date_col].unique():
            mask = self.keys[self.date_col] == date
            self._date_to_indices[date] = np.where(mask)[0]
    
    def get_indices_by_dates(self, dates: List[int]) -> np.ndarray:
        """根据日期列表获取行索引 - O(1) 复杂度"""
        indices_list = []
        for date in dates:
            if date in self._date_to_indices:
                indices_list.append(self._date_to_indices[date])
        return np.concatenate(indices_list) if indices_list else np.array([], dtype=int)
    
    def take(self, indices: np.ndarray) -> "SplitView":
        """根据索引提取子集 - 返回 SplitView"""
        return SplitView(
            indices=indices,
            X=self.X_full[indices],
            y=self.y_full[indices],
            keys=self.keys.iloc[indices].reset_index(drop=True),
            feature_names=self.feature_name_list.copy(),
            label_names=self.label_name_list.copy(),
            extra=self.extra.iloc[indices].reset_index(drop=True) if self.extra is not None else None,
        )
    
    @property
    def n_samples(self) -> int:
        return len(self.keys)
    
    @property
    def n_features(self) -> int:
        return len(self.feature_name_list)
    
    @property
    def dates(self) -> np.ndarray:
        """获取所有唯一日期"""
        return self.keys[self.date_col].unique()
    
    def get_feature_names_hash(self) -> str:
        """获取特征名哈希，防止错位"""
        return hashlib.md5(",".join(self.feature_name_list).encode()).hexdigest()[:8]


@dataclass
class SplitView:
    """
    单个数据集视图 (train/val/test 之一)
    """
    indices: np.ndarray
    X: np.ndarray
    y: np.ndarray
    keys: pd.DataFrame
    feature_names: List[str]
    label_names: List[str]
    extra: Optional[pd.DataFrame] = None
    group: Optional[pd.DataFrame] = None
    
    @property
    def n_samples(self) -> int:
        return len(self.indices)
    
    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1
    
    def get_feature_names_hash(self) -> str:
        """获取特征名哈希，用于防止错位"""
        return hashlib.md5(",".join(self.feature_names).encode()).hexdigest()[:8]


@dataclass
class DataBundle:
    """
    数据包 - 单个模式的数据集合
    """
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    feature_names: Optional[List[str]] = None


@dataclass
class SplitData:
    """
    切分数据 - 完整的 train/val/test 数据集合
    """
    train: DataBundle
    val: DataBundle
    test: DataBundle
    split_spec: Any  # SplitSpec，避免循环导入


@dataclass
class SplitViews:
    """
    完整的 split 视图集合
    """
    train: SplitView
    val: SplitView
    test: SplitView
    split_spec: Any  # SplitSpec，避免循环导入


@dataclass
class ProcessedViews:
    """
    DataProcessor 处理后的视图
    """
    train: SplitView
    val: SplitView
    test: SplitView
    split_spec: SplitSpec
    feature_mask: Optional[np.ndarray] = None
    transform_state: Optional["TransformState"] = None
    
    def get(self, mode: str) -> SplitView:
        return {"train": self.train, "val": self.val, "test": self.test}[mode]


@dataclass
class TransformState:
    """
    变换状态 - 可序列化
    """
    statistics: Dict[str, Any] = field(default_factory=dict)
    feature_mask: Optional[np.ndarray] = None
    feature_weights: Optional[np.ndarray] = None
    
    def save(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> "TransformState":
        with open(path, 'rb') as f:
            return pickle.load(f)




