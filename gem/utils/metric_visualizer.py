"""
Daily metric visualization.

Input: `daily_metric_series.csv` produced by ExperimentManager.
The file is expected to contain:
- split_id
- mode
- metric
- date (YYYYMMDD int)
- value

Visualization pipeline:
1) Aggregate by day with mean across splits for same date/mode/metric.
2) Export tidy daily metrics.
3) Plot daily curves with date x-axis.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

log = logging.getLogger(__name__)

MODE_ORDER: Tuple[str, ...] = ("train", "val", "test")
MODE_COLOR = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
}

DEFAULT_METRIC_PRIORITY: Tuple[str, ...] = (
    "daily_top_ret",
    "daily_top_ret_std",
    "daily_top_ret_relative_improve_pct",
    "daily_ic",
    "daily_icir_expanding",
    "daily_model_benchmark_corr",
    "daily_benchmark_top_ret",
)

_INVALID_FILENAME_CHARS = re.compile(r"[^0-9A-Za-z._-]+")
_EPS = 1e-8


class MetricsVisualizer:
    """Visualize daily metrics aggregated across splits."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        mode_order: Sequence[str] = MODE_ORDER,
    ):
        self.figsize = figsize
        self.mode_order = tuple(mode_order)
        self._daily_df: Optional[pl.DataFrame] = None

    @property
    def daily_df(self) -> Optional[pl.DataFrame]:
        return self._daily_df

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
    def _format_date_axis(ax) -> None:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    @staticmethod
    def _safe_metric_filename(metric_name: str) -> str:
        safe = _INVALID_FILENAME_CHARS.sub("_", metric_name).strip("_")
        return safe or "metric"

    @staticmethod
    def _extract_datetime_xy(df: pl.DataFrame) -> Tuple[List[datetime], List[float]]:
        dates: List[datetime] = []
        values: List[float] = []
        for row in df.iter_rows(named=True):
            date_raw = row.get("date")
            value_raw = row.get("value")
            if date_raw is None or value_raw is None:
                continue
            try:
                dt = datetime.strptime(str(int(date_raw)), "%Y%m%d")
                val = float(value_raw)
            except (TypeError, ValueError):
                continue
            dates.append(dt)
            values.append(val)
        return dates, values

    def load_from_daily_series_csv(self, csv_path: Path | str) -> Optional[pl.DataFrame]:
        path = Path(csv_path)
        if not path.exists():
            log.warning("Daily metric series file not found: %s", path)
            self._daily_df = None
            return None

        raw = pl.read_csv(path)
        required = {"mode", "metric", "date", "value"}
        if not required.issubset(set(raw.columns)):
            log.warning(
                "Daily metric series file missing required columns: %s",
                sorted(required - set(raw.columns)),
            )
            self._daily_df = None
            return None

        df = raw.select(
            [col for col in ("split_id", "mode", "metric", "date", "value") if col in raw.columns]
        ).with_columns(
            pl.col("mode").cast(pl.Utf8),
            pl.col("metric").cast(pl.Utf8),
            pl.col("value").cast(pl.Float64, strict=False),
            pl.col("date")
            .map_elements(self._normalize_date_int, return_dtype=pl.Int64)
            .alias("date"),
        ).drop_nulls(["mode", "metric", "date", "value"])

        if df.is_empty():
            self._daily_df = None
            return None

        if "split_id" in df.columns:
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

        daily = grouped.with_columns(
            pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .alias("date_dt")
        ).select(["date", "date_dt", "mode", "metric", "value", "n_split"]).sort(
            ["metric", "mode", "date"]
        )

        self._daily_df = self._append_derived_metrics(daily)
        return self._daily_df

    def _append_derived_metrics(self, daily_df: pl.DataFrame) -> pl.DataFrame:
        frames: List[pl.DataFrame] = [daily_df]

        # Derive expanding std for daily top return.
        top_ret_df = daily_df.filter(pl.col("metric") == "daily_top_ret").sort(
            ["mode", "date"]
        )
        if not top_ret_df.is_empty():
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
                ).select(["date", "date_dt", "mode", "metric", "value", "n_split"])
                top_ret_std_parts.append(part)
            if top_ret_std_parts:
                frames.append(pl.concat(top_ret_std_parts, how="vertical"))

        # Derive daily relative improve pct from daily model and benchmark returns.
        model_df = top_ret_df.rename(
            {"value": "model_value", "n_split": "model_n_split"}
        )
        bench_df = daily_df.filter(pl.col("metric") == "daily_benchmark_top_ret").rename(
            {"value": "bench_value", "n_split": "bench_n_split"}
        )
        if not model_df.is_empty() and not bench_df.is_empty():
            rel_df = (
                model_df.join(
                    bench_df,
                    on=["date", "date_dt", "mode"],
                    how="inner",
                )
                .with_columns(
                    pl.when(pl.col("bench_value").abs() > _EPS)
                    .then((pl.col("model_value") - pl.col("bench_value")) / pl.col("bench_value").abs() * 100.0)
                    .otherwise(0.0)
                    .alias("value"),
                    pl.max_horizontal("model_n_split", "bench_n_split").alias("n_split"),
                    pl.lit("daily_top_ret_relative_improve_pct").alias("metric"),
                )
                .select(["date", "date_dt", "mode", "metric", "value", "n_split"])
            )
            if not rel_df.is_empty():
                frames.append(rel_df)

        # Derive expanding ICIR from daily IC.
        ic_df = daily_df.filter(pl.col("metric") == "daily_ic").sort(["mode", "date"])
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
                icir = np.where(std > _EPS, means / (std + _EPS), 0.0)
                part = mode_df.with_columns(
                    pl.Series("value", icir),
                    pl.lit("daily_icir_expanding").alias("metric"),
                ).select(["date", "date_dt", "mode", "metric", "value", "n_split"])
                icir_parts.append(part)
            if icir_parts:
                frames.append(pl.concat(icir_parts, how="vertical"))

        return pl.concat(frames, how="vertical").sort(["metric", "mode", "date"])

    def available_metrics(self) -> List[str]:
        if self._daily_df is None or self._daily_df.is_empty():
            return []
        return sorted(self._daily_df["metric"].unique().to_list())

    def _resolve_metric_names(self, metric_names: Optional[Sequence[str]]) -> List[str]:
        all_metrics = self.available_metrics()
        if not all_metrics:
            return []
        daily_metrics = [metric for metric in all_metrics if metric.startswith("daily_")]
        available = set(daily_metrics)

        if metric_names:
            resolved = list(dict.fromkeys(str(metric) for metric in metric_names))
            non_daily = [metric for metric in resolved if not metric.startswith("daily_")]
            if non_daily:
                raise ValueError(
                    f"Unsupported metric_names (daily-only): {non_daily}. "
                    f"Available daily metrics: {sorted(available)}"
                )
            missing = [metric for metric in resolved if metric not in available]
            if missing:
                raise ValueError(
                    f"Unsupported metric_names: {missing}. "
                    f"Available daily metrics: {sorted(available)}"
                )
            return resolved

        priority = [name for name in DEFAULT_METRIC_PRIORITY if name in available]
        remaining = sorted(name for name in daily_metrics if name not in set(priority))
        return priority + remaining

    def export_metric_data_csv(
        self,
        output_path: Path | str,
        metric_names: Optional[Sequence[str]] = None,
    ) -> Optional[Path]:
        if self._daily_df is None or self._daily_df.is_empty():
            return None

        selected = self._resolve_metric_names(metric_names)
        if not selected:
            return None

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        (
            self._daily_df
            .filter(pl.col("metric").is_in(selected))
            .with_columns(pl.col("date_dt").dt.strftime("%Y-%m-%d").alias("date_str"))
            .select(["date", "date_str", "mode", "metric", "value", "n_split"])
            .sort(["metric", "mode", "date"])
            .write_csv(path)
        )
        return path

    def plot_metric_by_day(
        self,
        metric_name: str,
        output_path: Optional[Path | str] = None,
        show: bool = False,
    ):
        if self._daily_df is None or self._daily_df.is_empty():
            return None

        metric_df = self._daily_df.filter(pl.col("metric") == metric_name)
        if metric_df.is_empty():
            return None

        fig, ax = plt.subplots(figsize=self.figsize)
        has_curve = False
        for mode in self.mode_order:
            mode_df = metric_df.filter(pl.col("mode") == mode).sort("date")
            if mode_df.is_empty():
                continue
            x, y = self._extract_datetime_xy(mode_df)
            if not x:
                continue
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.6,
                markersize=3.5,
                label=mode,
                color=MODE_COLOR.get(mode),
            )
            has_curve = True

        if not has_curve:
            plt.close(fig)
            return None

        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} by day (mean over splits)")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        self._format_date_axis(ax)
        fig.autofmt_xdate()
        plt.tight_layout()

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_overview(
        self,
        output_path: Optional[Path | str] = None,
        metric_names: Optional[Sequence[str]] = None,
        show: bool = False,
        ncols: int = 2,
    ):
        if self._daily_df is None or self._daily_df.is_empty():
            return None
        selected = self._resolve_metric_names(metric_names)
        if not selected:
            return None

        ncols = max(1, min(ncols, len(selected)))
        nrows = int(math.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
        flat_axes = [ax for row in axes for ax in row]

        for idx, metric_name in enumerate(selected):
            ax = flat_axes[idx]
            metric_df = self._daily_df.filter(pl.col("metric") == metric_name)
            for mode in self.mode_order:
                mode_df = metric_df.filter(pl.col("mode") == mode).sort("date")
                if mode_df.is_empty():
                    continue
                x, y = self._extract_datetime_xy(mode_df)
                if not x:
                    continue
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.4,
                    markersize=2.8,
                    label=mode,
                    color=MODE_COLOR.get(mode),
                )
            ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle="--")
            ax.set_title(metric_name)
            ax.grid(alpha=0.2)
            if idx % ncols == 0:
                ax.set_ylabel("Value")
            if idx // ncols == nrows - 1:
                ax.set_xlabel("Date")
            self._format_date_axis(ax)
            ax.legend(loc="best", fontsize=8)

        for idx in range(len(selected), len(flat_axes)):
            fig.delaxes(flat_axes[idx])

        fig.autofmt_xdate()
        plt.tight_layout()
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_distribution(
        self,
        output_path: Optional[Path | str] = None,
        metric_names: Optional[Sequence[str]] = None,
        show: bool = False,
        ncols: int = 2,
    ):
        if self._daily_df is None or self._daily_df.is_empty():
            return None
        selected = self._resolve_metric_names(metric_names)
        if not selected:
            return None

        ncols = max(1, min(ncols, len(selected)))
        nrows = int(math.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
        flat_axes = [ax for row in axes for ax in row]

        for idx, metric_name in enumerate(selected):
            ax = flat_axes[idx]
            metric_df = self._daily_df.filter(pl.col("metric") == metric_name)
            data = []
            labels = []
            for mode in self.mode_order:
                mode_df = metric_df.filter(pl.col("mode") == mode)
                if mode_df.is_empty():
                    continue
                data.append(mode_df["value"].to_numpy())
                labels.append(mode)

            if not data:
                ax.set_title(f"{metric_name} (no data)")
                ax.axis("off")
                continue

            ax.boxplot(data, labels=labels)
            ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle="--")
            ax.set_title(metric_name)
            ax.set_ylabel("Daily value")
            ax.grid(alpha=0.2, axis="y")

        for idx in range(len(selected), len(flat_axes)):
            fig.delaxes(flat_axes[idx])

        plt.tight_layout()
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def create_all_plots(
        self,
        output_dir: Path | str,
        metric_names: Optional[Sequence[str]] = None,
        show: bool = False,
        export_csv: bool = True,
        overview: bool = True,
        distribution: bool = True,
        per_metric: bool = True,
    ) -> List[Path]:
        if self._daily_df is None or self._daily_df.is_empty():
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = self._resolve_metric_names(metric_names)
        if not selected:
            return []

        saved_paths: List[Path] = []
        if export_csv:
            csv_path = self.export_metric_data_csv(output_dir / "metrics_data.csv", selected)
            if csv_path is not None:
                saved_paths.append(csv_path)

        if overview:
            path = output_dir / "metrics_overview.png"
            if self.plot_overview(path, selected, show=show) is not None:
                saved_paths.append(path)

        if distribution:
            path = output_dir / "metrics_distribution.png"
            if self.plot_distribution(path, selected, show=show) is not None:
                saved_paths.append(path)

        if per_metric:
            for metric_name in selected:
                safe_name = self._safe_metric_filename(metric_name)
                path = output_dir / f"metric_{safe_name}_by_day.png"
                if self.plot_metric_by_day(metric_name, path, show=show) is not None:
                    saved_paths.append(path)

        return saved_paths
