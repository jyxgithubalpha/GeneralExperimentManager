"""
Core data structures shared across data, method and experiment modules.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ..utils.hash_utils import hash_feature_names


@dataclass
class SplitSpec:
    split_id: int
    train_date_list: List[int]
    val_date_list: List[int]
    test_date_list: List[int]
    anchor_time: Optional[int] = None

    def get_all_dates_range(self) -> Tuple[int, int]:
        all_dates = self.train_date_list + self.val_date_list + self.test_date_list
        if not all_dates:
            raise ValueError(f"Split {self.split_id} has no dates in train/val/test.")
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
    layout: str = "tabular"
    date_col: str = "date"
    code_col: str = "code"
    pivot: Optional[str] = None
    index_col: Optional[str] = None
    value_col: Optional[str] = None
    rename_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    X_source_list: List[str] = field(default_factory=list)
    y_source_list: List[str] = field(default_factory=list)
    extra_source_list: List[str] = field(default_factory=list)
    key_cols: List[str] = field(default_factory=list)
    group_col: Optional[List[str]] = None


@dataclass
class GlobalStore:
    keys: pl.DataFrame
    X_full: np.ndarray
    y_full: np.ndarray
    feature_name_list: List[str]
    label_name_list: List[str]
    date_col: str = "date"
    code_col: str = "code"
    extra: Optional[pl.DataFrame] = None

    _date_to_indices: Dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        self._build_date_index()

    def _validate(self) -> None:
        n_samples = self.keys.height

        if self.X_full.shape[0] != n_samples:
            raise ValueError(
                f"X_full row count ({self.X_full.shape[0]}) does not match keys ({n_samples})."
            )
        if self.y_full.shape[0] != n_samples:
            raise ValueError(
                f"y_full row count ({self.y_full.shape[0]}) does not match keys ({n_samples})."
            )
        if self.X_full.ndim != 2:
            raise ValueError(f"X_full must be 2D, got shape={self.X_full.shape}.")

        if self.y_full.ndim == 1:
            self.y_full = self.y_full.reshape(-1, 1)
        elif self.y_full.ndim != 2:
            raise ValueError(f"y_full must be 1D or 2D, got shape={self.y_full.shape}.")

        if self.X_full.shape[1] != len(self.feature_name_list):
            raise ValueError(
                f"Feature column count ({self.X_full.shape[1]}) does not match "
                f"feature_name_list ({len(self.feature_name_list)})."
            )
        if self.y_full.shape[1] != len(self.label_name_list):
            raise ValueError(
                f"Label column count ({self.y_full.shape[1]}) does not match "
                f"label_name_list ({len(self.label_name_list)})."
            )

        if self.date_col not in self.keys.columns:
            raise ValueError(f"Missing required key column '{self.date_col}'.")
        if self.code_col not in self.keys.columns:
            raise ValueError(f"Missing required key column '{self.code_col}'.")

    def _build_date_index(self) -> None:
        self._date_to_indices.clear()
        date_arr = self.keys[self.date_col].to_numpy()
        for date in np.unique(date_arr):
            self._date_to_indices[int(date)] = np.flatnonzero(date_arr == date)

    def get_indices_by_dates(self, dates: List[int]) -> np.ndarray:
        indices_list: List[np.ndarray] = []
        for date in dates:
            day_indices = self._date_to_indices.get(int(date))
            if day_indices is not None:
                indices_list.append(day_indices)

        if not indices_list:
            return np.array([], dtype=np.int64)
        return np.concatenate(indices_list)

    def take(self, indices: np.ndarray) -> "SplitView":
        idx = np.asarray(indices, dtype=np.int64)
        idx_list = idx.tolist()
        return SplitView(
            indices=idx,
            X=self.X_full[idx],
            y=self.y_full[idx],
            keys=self.keys[idx_list],
            feature_names=self.feature_name_list.copy(),
            label_names=self.label_name_list.copy(),
            extra=self.extra[idx_list] if self.extra is not None else None,
        )

    @property
    def n_samples(self) -> int:
        return self.keys.height

    @property
    def n_features(self) -> int:
        return len(self.feature_name_list)

    @property
    def dates(self) -> np.ndarray:
        return self.keys[self.date_col].unique().sort().to_numpy()

    def get_feature_names_hash(self) -> str:
        return hash_feature_names(self.feature_name_list)


@dataclass
class SplitView:
    """Single split view: train / val / test."""

    indices: np.ndarray
    X: np.ndarray
    y: np.ndarray
    keys: pl.DataFrame
    feature_names: List[str]
    label_names: List[str]
    extra: Optional[pl.DataFrame] = None
    group: Optional[pl.DataFrame] = None

    @property
    def n_samples(self) -> int:
        return len(self.indices)

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1

    def get_feature_names_hash(self) -> str:
        return hash_feature_names(self.feature_names)


@dataclass
class DataBundle:
    X: np.ndarray
    y: np.ndarray
    meta: pl.DataFrame
    feature_names: Optional[List[str]] = None


@dataclass
class SplitData:
    train: DataBundle
    val: DataBundle
    test: DataBundle
    split_spec: Any


@dataclass
class SplitViews:
    train: SplitView
    val: SplitView
    test: SplitView
    split_spec: Any


@dataclass
class ProcessedViews:
    train: SplitView
    val: SplitView
    test: SplitView
    split_spec: SplitSpec
    feature_mask: Optional[np.ndarray] = None
    transform_state: Optional["TransformState"] = None

    def get(self, mode: str) -> SplitView:
        try:
            return {"train": self.train, "val": self.val, "test": self.test}[mode]
        except KeyError as exc:
            raise ValueError(
                f"Unknown mode '{mode}', expected one of train/val/test."
            ) from exc


@dataclass
class TransformState:
    statistics: Dict[str, Any] = field(default_factory=dict)
    feature_mask: Optional[np.ndarray] = None
    feature_weights: Optional[np.ndarray] = None

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "TransformState":
        with open(path, "rb") as f:
            return pickle.load(f)
