"""
Report generation for experiment results.
"""


import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from .configs import ExperimentConfig
from .results import SplitResult

log = logging.getLogger(__name__)
_EPS = 1e-8
_REL_IMPROVE_DENOM_FLOOR = 0.01
_ICIR_MIN_PERIODS = 20


class ReportGenerator:
    """Generate experiment reports from split results."""

    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config

    def generate(
        self,
        results: Dict[int, SplitResult],
        output_dir: Path,
    ) -> None:
        """Generate all report files."""
        rows = []
        series_rows = []
        for split_id, result in sorted(results.items()):
            status = "failed" if result.failed else ("skipped" if result.skipped else "success")
            row = {
                "split_id": split_id,
                "status": status,
                "skipped": result.skipped,
                "failed": result.failed,
                "skip_reason": result.skip_reason,
                "error_message": result.error_message,
                "error_trace_path": result.error_trace_path,
            }
            if result.metrics:
                row.update(result.metrics)
            if result.metric_series_rows:
                for item in result.metric_series_rows:
                    series_rows.append(
                        {
                            "split_id": split_id,
                            "mode": item.get("mode"),
                            "metric": item.get("metric"),
                            "date": item.get("date"),
                            "value": item.get("value"),
                        }
                    )
            rows.append(row)

        df = pl.DataFrame(rows) if rows else pl.DataFrame(
            schema={
                "split_id": pl.Int64,
                "status": pl.Utf8,
                "skipped": pl.Boolean,
                "failed": pl.Boolean,
                "skip_reason": pl.Utf8,
                "error_message": pl.Utf8,
                "error_trace_path": pl.Utf8,
            }
        )

        csv_path = output_dir / "results_summary.csv"
        df.write_csv(csv_path)
        log.info("  - Saved: %s", csv_path)

        if series_rows:
            series_df = self._append_derived_daily_metrics(
                self._aggregate_daily_metric_series(series_rows)
            )
            series_csv_path = output_dir / "daily_metric_series.csv"
            series_df.write_csv(series_csv_path)
            log.info("  - Saved: %s", series_csv_path)

        config_path = output_dir / "config.json"
        config_dict = {
            "name": self.experiment_config.name,
            "seed": self.experiment_config.seed,
            "n_trials": self.experiment_config.n_trials,
            "parallel_trials": self.experiment_config.parallel_trials,
            "use_ray_tune": self.experiment_config.use_ray_tune,
            "state_policy_mode": self.experiment_config.state_policy.mode,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)
        log.info("  - Saved: %s", config_path)

        n_skipped = sum(1 for item in results.values() if item.skipped)
        n_failed = sum(1 for item in results.values() if item.failed)
        n_success = len(results) - n_skipped - n_failed
        log.info("  Splits: %d success, %d skipped, %d failed", n_success, n_skipped, n_failed)

    @staticmethod
    def _empty_daily_metric_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "date": pl.Int64,
                "mode": pl.Utf8,
                "metric": pl.Utf8,
                "value": pl.Float64,
                "n_split": pl.Int64,
                "is_derived": pl.Boolean,
                "source_metric": pl.Utf8,
            }
        )

    @staticmethod
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

    @staticmethod
    def _aggregate_daily_metric_series(series_rows: List[Dict[str, object]]) -> pl.DataFrame:
        if not series_rows:
            return ReportGenerator._empty_daily_metric_df()

        raw = pl.DataFrame(series_rows)
        required = {"mode", "metric", "date", "value"}
        if not required.issubset(set(raw.columns)):
            missing = sorted(required - set(raw.columns))
            log.warning("Skip metric aggregation because required columns are missing: %s", missing)
            return ReportGenerator._empty_daily_metric_df()

        has_split_id = "split_id" in raw.columns
        selected_cols = [col for col in ("split_id", "mode", "metric", "date", "value") if col in raw.columns]
        df = raw.select(selected_cols).with_columns(
            pl.col("mode").cast(pl.Utf8, strict=False),
            pl.col("metric").cast(pl.Utf8, strict=False),
            pl.col("date")
            .map_elements(ReportGenerator._normalize_date_int, return_dtype=pl.Int64)
            .alias("date"),
            pl.col("value").cast(pl.Float64, strict=False),
        )
        if has_split_id:
            df = df.with_columns(pl.col("split_id").cast(pl.Int64, strict=False))

        drop_cols = ["mode", "metric", "date", "value"] + (["split_id"] if has_split_id else [])
        pre_drop_count = df.height
        df = df.drop_nulls(drop_cols).filter(pl.col("value").is_finite())
        filtered_count = pre_drop_count - df.height
        if filtered_count > 0:
            log.warning("Dropped %d invalid metric series rows before aggregation.", filtered_count)

        if df.is_empty():
            return ReportGenerator._empty_daily_metric_df()

        if has_split_id:
            grouped = (
                df.group_by(["date", "mode", "metric"])
                .agg(
                    pl.col("value").mean().alias("value"),
                    pl.col("split_id").n_unique().alias("n_split"),
                )
            )
        else:
            grouped = (
                df.group_by(["date", "mode", "metric"])
                .agg(
                    pl.col("value").mean().alias("value"),
                    pl.len().alias("n_split"),
                )
            )

        return (
            grouped.with_columns(
                pl.lit(False).alias("is_derived"),
                pl.lit(None, dtype=pl.Utf8).alias("source_metric"),
            )
            .select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
            .sort(["metric", "mode", "date"])
        )

    @staticmethod
    def _append_derived_daily_metrics(daily_df: pl.DataFrame) -> pl.DataFrame:
        if daily_df.is_empty():
            return daily_df

        rel_floor_hit_count = 0
        icir_min_periods_null_count = 0

        base = (
            daily_df.with_columns(
                pl.col("is_derived").cast(pl.Boolean, strict=False).fill_null(False),
                pl.col("source_metric").cast(pl.Utf8, strict=False),
            )
            .select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
            .sort(["metric", "mode", "date"])
        )

        frames: List[pl.DataFrame] = [base]
        existing_metrics = set(base["metric"].unique().to_list())

        top_ret_df = base.filter(pl.col("metric") == "daily_top_ret").sort(["mode", "date"])
        if not top_ret_df.is_empty() and "daily_top_ret_std" not in existing_metrics:
            top_ret_std_parts: List[pl.DataFrame] = []
            for mode in top_ret_df["mode"].unique().to_list():
                mode_df = top_ret_df.filter(pl.col("mode") == mode).sort("date")
                if mode_df.is_empty():
                    continue
                vals = mode_df["value"].to_numpy()
                n = vals.shape[0]
                if n == 0:
                    continue
                cumsum = np.cumsum(vals)
                cumsq = np.cumsum(vals * vals)
                counts = np.arange(1, n + 1, dtype=np.float64)
                means = cumsum / counts
                var = np.maximum(cumsq / counts - means * means, 0.0)
                std = np.sqrt(var)
                part = mode_df.with_columns(
                    pl.Series("value", std),
                    pl.lit("daily_top_ret_std").alias("metric"),
                    pl.lit(True).alias("is_derived"),
                    pl.lit("daily_top_ret").alias("source_metric"),
                ).select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
                top_ret_std_parts.append(part)
            if top_ret_std_parts:
                frames.append(pl.concat(top_ret_std_parts, how="vertical"))

        if "daily_top_ret_relative_improve_pct" not in existing_metrics:
            model_df = top_ret_df.rename({"value": "model_value", "n_split": "model_n_split"})
            bench_df = base.filter(pl.col("metric") == "daily_benchmark_top_ret").rename(
                {"value": "bench_value", "n_split": "bench_n_split"}
            )
            if not model_df.is_empty() and not bench_df.is_empty():
                joined = model_df.join(
                    bench_df,
                    on=["date", "mode"],
                    how="inner",
                )
                rel_floor_hit_count += joined.filter(
                    pl.col("bench_value").abs() < _REL_IMPROVE_DENOM_FLOOR
                ).height
                rel_df = (
                    joined.with_columns(
                        pl.max_horizontal(
                            pl.col("bench_value").abs(),
                            pl.lit(_REL_IMPROVE_DENOM_FLOOR),
                        ).alias("denom"),
                    )
                    .with_columns(
                        ((pl.col("model_value") - pl.col("bench_value")) / pl.col("denom")).alias("value"),
                        pl.max_horizontal("model_n_split", "bench_n_split").alias("n_split"),
                        pl.lit("daily_top_ret_relative_improve_pct").alias("metric"),
                        pl.lit(True).alias("is_derived"),
                        pl.lit("daily_top_ret,daily_benchmark_top_ret").alias("source_metric"),
                    )
                    .drop("denom")
                    .select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
                )
                if not rel_df.is_empty():
                    frames.append(rel_df)

        if "daily_icir_expanding" not in existing_metrics:
            ic_df = base.filter(pl.col("metric") == "daily_ic").sort(["mode", "date"])
            if not ic_df.is_empty():
                icir_parts: List[pl.DataFrame] = []
                for mode in ic_df["mode"].unique().to_list():
                    mode_df = ic_df.filter(pl.col("mode") == mode).sort("date")
                    if mode_df.is_empty():
                        continue
                    vals = mode_df["value"].to_numpy()
                    n = vals.shape[0]
                    if n == 0:
                        continue
                    cumsum = np.cumsum(vals)
                    cumsq = np.cumsum(vals * vals)
                    counts = np.arange(1, n + 1, dtype=np.float64)
                    means = cumsum / counts
                    var = np.maximum(cumsq / counts - means * means, 0.0)
                    std = np.sqrt(var)
                    icir: List[Optional[float]] = []
                    for idx in range(n):
                        if counts[idx] < _ICIR_MIN_PERIODS:
                            icir_min_periods_null_count += 1
                            icir.append(None)
                        elif std[idx] > _EPS:
                            icir.append(float(means[idx] / (std[idx] + _EPS)))
                        else:
                            icir.append(None)
                    part = mode_df.with_columns(
                        pl.Series("value", icir),
                        pl.lit("daily_icir_expanding").alias("metric"),
                        pl.lit(True).alias("is_derived"),
                        pl.lit("daily_ic").alias("source_metric"),
                    ).select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
                    icir_parts.append(part)
                if icir_parts:
                    frames.append(pl.concat(icir_parts, how="vertical"))

        if rel_floor_hit_count > 0:
            log.warning(
                "Applied denominator floor %.4f for daily_top_ret_relative_improve_pct on %d rows.",
                _REL_IMPROVE_DENOM_FLOOR,
                rel_floor_hit_count,
            )
        if icir_min_periods_null_count > 0:
            log.info(
                "Set daily_icir_expanding to null for %d rows due to min_periods=%d.",
                icir_min_periods_null_count,
                _ICIR_MIN_PERIODS,
            )

        return pl.concat(frames, how="vertical").sort(["metric", "mode", "date"])
