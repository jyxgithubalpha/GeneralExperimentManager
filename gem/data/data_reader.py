"""
Data readers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
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
        return {
            source_name: self.read(source_spec, date_start, date_end)
            for source_name, source_spec in source_spec_dict.items()
        }

    @staticmethod
    def _standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
        if "Code" in df.columns and "code" not in df.columns:
            df = df.rename({"Code": "code"})

        if "date" in df.columns and "code" in df.columns:
            other_cols = [c for c in df.columns if c not in ("date", "code")]
            df = df.select(["date", "code", *other_cols])

        if "date" in df.columns:
            df = df.with_columns(pl.col("date").cast(pl.Int32))
        if "code" in df.columns:
            df = df.with_columns(pl.col("code").cast(pl.Utf8))
        return df


class FeatherReader(DataReader):
    _PIVOT_FROM_INDEX = {"ret", "liquidity"}
    _PIVOT_FROM_DATE = {
        "score",
        "benchmark",
        "bench1",
        "bench2",
        "bench3",
        "bench4",
        "bench5",
        "bench6",
    }

    def _filter_date_range(self, df: pl.DataFrame, date_start: int, date_end: int) -> pl.DataFrame:
        if "date" not in df.columns:
            return df
        return df.filter((pl.col("date") >= date_start) & (pl.col("date") <= date_end))

    def _process_dataframe(
        self,
        df: pl.DataFrame,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
    ) -> pl.DataFrame:
        source_name = source_spec.name

        if source_name == "fac":
            df = df.with_columns(pl.col("date").cast(pl.Int32))
            df = self._filter_date_range(df, date_start, date_end)

        elif source_name in self._PIVOT_FROM_INDEX:
            df = df.unpivot(
                index=["index"],
                variable_name="code",
                value_name=f"{source_name}_value",
            )
            df = df.rename({"index": "date"})
            df = df.with_columns(pl.col("date").cast(pl.Int32))
            df = self._filter_date_range(df, date_start, date_end)

        elif source_name in self._PIVOT_FROM_DATE:
            df = df.unpivot(
                index=["date"],
                variable_name="code",
                value_name=f"{source_name}_value",
            )
            df = df.with_columns(pl.col("date").cast(pl.Int32))
            df = self._filter_date_range(df, date_start, date_end)

        return self._standardize_columns(df)

    def read(
        self,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
        filters: Optional[Any] = None,
    ) -> pl.DataFrame:
        path = Path(source_spec.path)
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {path}")

        feather_reader = pa.ipc.open_file(str(path))
        batches = []

        for batch_idx in range(feather_reader.num_record_batches):
            batch = feather_reader.get_record_batch(batch_idx)
            batch_df = pl.from_arrow(batch)
            processed_batch = self._process_dataframe(batch_df, source_spec, date_start, date_end)
            if processed_batch.height > 0:
                batches.append(processed_batch)

        if not batches:
            return pl.DataFrame()

        return pl.concat(batches)
