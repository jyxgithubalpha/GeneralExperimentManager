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
    def _standardize_columns(
        df: pl.DataFrame,
        *,
        date_col: str = "date",
        code_col: str = "code",
        rename_map: Optional[Dict[str, str]] = None,
    ) -> pl.DataFrame:
        rename_ops: Dict[str, str] = {}
        if rename_map:
            rename_ops.update(rename_map)
        if "Code" in df.columns and code_col == "code" and "code" not in df.columns:
            rename_ops["Code"] = "code"
        if date_col in df.columns and date_col != "date":
            rename_ops[date_col] = "date"
        if code_col in df.columns and code_col != "code":
            rename_ops[code_col] = "code"
        if rename_ops:
            df = df.rename(rename_ops)

        if "date" in df.columns and "code" in df.columns:
            other_cols = [c for c in df.columns if c not in ("date", "code")]
            df = df.select(["date", "code", *other_cols])

        if "date" in df.columns:
            df = df.with_columns(pl.col("date").cast(pl.Int32))
        if "code" in df.columns:
            df = df.with_columns(pl.col("code").cast(pl.Utf8))
        return df


class FeatherReader(DataReader):
    def _filter_date_range(self, df: pl.DataFrame, date_start: int, date_end: int) -> pl.DataFrame:
        if "date" not in df.columns:
            return df
        return df.filter((pl.col("date") >= date_start) & (pl.col("date") <= date_end))

    @staticmethod
    def _resolve_value_col(source_spec: SourceSpec) -> str:
        if source_spec.value_col:
            return source_spec.value_col
        return f"{source_spec.name}_value"

    def _empty_frame_for_source(self, source_spec: SourceSpec) -> pl.DataFrame:
        schema: Dict[str, pl.DataType] = {
            "date": pl.Int32,
            "code": pl.Utf8,
        }
        value_col = source_spec.value_col
        if value_col:
            schema[value_col] = pl.Float32
        return pl.DataFrame(schema=schema)

    def _process_dataframe(
        self,
        df: pl.DataFrame,
        source_spec: SourceSpec,
        date_start: int,
        date_end: int,
    ) -> pl.DataFrame:
        layout = (source_spec.layout or "tabular").lower()
        pivot = (source_spec.pivot or "").lower()
        date_col = source_spec.date_col or "date"
        code_col = source_spec.code_col or "code"
        value_col = self._resolve_value_col(source_spec)
        index_col = source_spec.index_col or "index"

        if layout == "wide":
            if pivot == "from_index":
                if index_col not in df.columns:
                    raise ValueError(
                        f"Source '{source_spec.name}' expects index_col '{index_col}' for pivot 'from_index'."
                    )
                df = df.unpivot(
                    index=[index_col],
                    variable_name=code_col,
                    value_name=value_col,
                )
                if index_col != date_col:
                    df = df.rename({index_col: date_col})
            elif pivot == "from_date":
                if date_col not in df.columns:
                    raise ValueError(
                        f"Source '{source_spec.name}' expects date_col '{date_col}' for pivot 'from_date'."
                    )
                df = df.unpivot(
                    index=[date_col],
                    variable_name=code_col,
                    value_name=value_col,
                )
            else:
                raise ValueError(
                    f"Source '{source_spec.name}' has layout='wide' but unsupported pivot='{source_spec.pivot}'. "
                    "Expected one of: from_index, from_date."
                )
        elif layout not in {"tabular", "long"}:
            raise ValueError(
                f"Unsupported layout '{source_spec.layout}' for source '{source_spec.name}'."
            )

        df = self._standardize_columns(
            df,
            date_col=date_col,
            code_col=code_col,
            rename_map=source_spec.rename_map,
        )
        return self._filter_date_range(df, date_start, date_end)

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
            batches.append(processed_batch)

        if not batches:
            return self._empty_frame_for_source(source_spec)

        return pl.concat(batches, how="vertical_relaxed")
