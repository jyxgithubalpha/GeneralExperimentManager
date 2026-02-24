from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

log = logging.getLogger(__name__)

REQUIRED_METRIC_COLUMNS = {
    "date",
    "mode",
    "metric",
    "value",
    "n_split",
    "is_derived",
    "source_metric",
}


def _normalize_date_int(raw_value: object) -> Optional[int]:
    if raw_value is None:
        return None
    try:
        date_int = int(raw_value)
    except (TypeError, ValueError):
        return None
    text = str(date_int)
    if len(text) != 8:
        return None
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return None
    return date_int


def load_metrics_data(csv_path: Path | str) -> Tuple[Optional[pl.DataFrame], Optional[str]]:
    path = Path(csv_path)
    if not path.exists():
        return None, f"Metric series file not found: {path}"

    try:
        raw = pl.read_csv(path)
    except Exception as exc:  # pragma: no cover - I/O error branch
        return None, f"Failed to read metric series CSV: {exc}"

    missing = sorted(REQUIRED_METRIC_COLUMNS - set(raw.columns))
    if missing:
        return None, f"Metric series CSV missing required columns: {missing}"

    df = raw.select(
        ["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"]
    ).with_columns(
        pl.col("date").map_elements(_normalize_date_int, return_dtype=pl.Int64).alias("date"),
        pl.col("mode").cast(pl.Utf8, strict=False),
        pl.col("metric").cast(pl.Utf8, strict=False),
        pl.col("value").cast(pl.Float64, strict=False),
        pl.col("n_split").cast(pl.Int64, strict=False),
        pl.col("is_derived").cast(pl.Boolean, strict=False),
        pl.col("source_metric").cast(pl.Utf8, strict=False),
    )

    pre_count = df.height
    df = (
        df.drop_nulls(["date", "mode", "metric", "value", "n_split", "is_derived"])
        .filter(pl.col("value").is_finite())
        .with_columns(
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .alias("date_dt")
        )
        .drop_nulls(["date_dt"])
        .sort(["metric", "mode", "date"])
    )
    filtered_count = pre_count - df.height
    if filtered_count > 0:
        log.warning("Dropped %d invalid rows from metric series CSV.", filtered_count)

    if df.is_empty():
        return None, "Metric series CSV has no valid rows after validation."
    return df, None

