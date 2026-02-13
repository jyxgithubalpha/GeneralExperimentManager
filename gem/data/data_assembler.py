"""
Data assemblers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

import numpy as np
import polars as pl

from .data_dataclasses import DatasetSpec, GlobalStore
from .utils import remove_quotes_from_list


class GlobalDataAssembler(ABC):
    @abstractmethod
    def assemble(self, source_dict: Dict[str, pl.DataFrame]) -> GlobalStore:
        """Assemble source frames into a `GlobalStore`."""


class FeatureAssembler(GlobalDataAssembler):
    """Assemble multiple source tables into one model-ready `GlobalStore`."""

    def __init__(self, dataset_spec: DatasetSpec):
        self.X_source_list = remove_quotes_from_list(dataset_spec.X_source_list)
        self.y_source_list = remove_quotes_from_list(dataset_spec.y_source_list)
        self.extra_source_list = remove_quotes_from_list(dataset_spec.extra_source_list)
        self.key_cols = remove_quotes_from_list(dataset_spec.key_cols)
        self.group_col = remove_quotes_from_list(dataset_spec.group_col)

        if not self.key_cols:
            raise ValueError("dataset_spec.key_cols must not be empty.")
        if not self.X_source_list:
            raise ValueError("dataset_spec.X_source_list must not be empty.")
        if not self.y_source_list:
            raise ValueError("dataset_spec.y_source_list must not be empty.")

    def _prefix_non_key_cols(self, df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        rename_map = {
            col: f"{prefix}__{col}"
            for col in df.columns
            if col not in self.key_cols
        }
        return df.rename(rename_map)

    def _validate_source_frame(self, df_name: str, df: pl.DataFrame) -> None:
        missing = [col for col in self.key_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Source '{df_name}' missing key columns: {missing}. "
                f"Available columns: {df.columns}"
            )

    def _require_sources(self, source_dict: Dict[str, pl.DataFrame], names: List[str]) -> List[pl.DataFrame]:
        missing = [name for name in names if name not in source_dict]
        if missing:
            raise KeyError(f"Missing required sources: {missing}")

        frames: List[pl.DataFrame] = []
        for name in names:
            frame = source_dict[name]
            self._validate_source_frame(name, frame)
            frames.append(self._prefix_non_key_cols(frame, name))
        return frames

    def _merge_sources(self, frames: List[pl.DataFrame]) -> pl.DataFrame:
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.join(df, on=self.key_cols, how="outer_coalesce")
        return merged

    def _to_matrix(self, df: pl.DataFrame, cols: List[str]) -> np.ndarray:
        if not cols:
            return np.empty((df.height, 0), dtype=np.float32)
        return df.select(cols).to_numpy().astype(np.float32)

    def assemble(self, source_dict: Dict[str, pl.DataFrame]) -> GlobalStore:
        X_df = self._merge_sources(self._require_sources(source_dict, self.X_source_list))
        y_df = self._merge_sources(self._require_sources(source_dict, self.y_source_list))

        extra_df: Optional[pl.DataFrame] = None
        if self.extra_source_list:
            extra_df = self._merge_sources(self._require_sources(source_dict, self.extra_source_list))

        keys = X_df.select(self.key_cols).unique(maintain_order=True)
        y_df = keys.join(y_df, on=self.key_cols, how="left")
        if extra_df is not None:
            extra_df = keys.join(extra_df, on=self.key_cols, how="left")

        feature_names = [c for c in X_df.columns if c not in self.key_cols]
        label_names = [c for c in y_df.columns if c not in self.key_cols]

        X_full = self._to_matrix(X_df, feature_names)
        y_full = self._to_matrix(y_df, label_names)

        return GlobalStore(
            keys=keys,
            X_full=X_full,
            y_full=y_full,
            feature_name_list=feature_names,
            label_name_list=label_names,
            extra=extra_df,
        )
