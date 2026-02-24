from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from ..experiment.experiment_dataclasses import MetricsVizConfig, VisualizationRenderConfig
from .contracts import VisualizationArtifact

log = logging.getLogger(__name__)

MODE_ORDER = ("train", "val", "test")
MODE_COLOR = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
}
DEFAULT_METRIC_PRIORITY = (
    "daily_top_ret",
    "daily_top_ret_std",
    "daily_top_ret_relative_improve_pct",
    "daily_ic",
    "daily_icir_expanding",
    "daily_model_benchmark_corr",
    "daily_benchmark_top_ret",
)
_INVALID_FILENAME_CHARS = re.compile(r"[^0-9A-Za-z._-]+")


def _build_artifact(
    name: str,
    kind: str,
    status: str,
    path: Optional[Path] = None,
    message: Optional[str] = None,
) -> VisualizationArtifact:
    return VisualizationArtifact(name=name, kind=kind, status=status, path=path, message=message)


def _format_date_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _safe_metric_filename(metric_name: str) -> str:
    safe = _INVALID_FILENAME_CHARS.sub("_", metric_name).strip("_")
    return safe or "metric"


def resolve_metric_names(df: pl.DataFrame, metric_names: Optional[Sequence[str]]) -> list[str]:
    all_metrics = sorted(df["metric"].unique().to_list())
    daily_metrics = [name for name in all_metrics if str(name).startswith("daily_")]
    available = set(daily_metrics)
    if metric_names:
        resolved = list(dict.fromkeys(str(item) for item in metric_names))
        non_daily = [name for name in resolved if not name.startswith("daily_")]
        if non_daily:
            raise ValueError(
                f"Unsupported metric_names (daily-only): {non_daily}. Available daily metrics: {sorted(available)}"
            )
        missing = [name for name in resolved if name not in available]
        if missing:
            raise ValueError(
                f"Unsupported metric_names: {missing}. Available daily metrics: {sorted(available)}"
            )
        return resolved
    priority = [name for name in DEFAULT_METRIC_PRIORITY if name in available]
    remaining = sorted(name for name in daily_metrics if name not in set(priority))
    return priority + remaining


def export_metrics_data_csv(
    df: pl.DataFrame,
    output_path: Path,
    metric_names: Sequence[str],
) -> Optional[Path]:
    if not metric_names:
        return None
    out_df = (
        df.filter(pl.col("metric").is_in(metric_names))
        .with_columns(pl.col("date_dt").dt.strftime("%Y-%m-%d").alias("date_str"))
        .select(
            [
                "date",
                "date_str",
                "mode",
                "metric",
                "value",
                "n_split",
                "is_derived",
                "source_metric",
            ]
        )
        .sort(["metric", "mode", "date"])
    )
    if out_df.is_empty():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(output_path)
    return output_path


def render_metric_by_day(
    df: pl.DataFrame,
    metric_name: str,
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
) -> Optional[Path]:
    metric_df = df.filter(pl.col("metric") == metric_name)
    if metric_df.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    has_curve = False
    for mode in MODE_ORDER:
        mode_df = metric_df.filter(pl.col("mode") == mode).sort("date")
        if mode_df.is_empty():
            continue
        x = mode_df["date_dt"].to_list()
        y = mode_df["value"].to_list()
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
    _format_date_axis(ax)
    fig.autofmt_xdate()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_cfg.dpi, bbox_inches="tight")
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def render_metrics_overview(
    df: pl.DataFrame,
    metric_names: Sequence[str],
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
    ncols: int = 2,
) -> Optional[Path]:
    if not metric_names:
        return None
    ncols = max(1, min(ncols, len(metric_names)))
    nrows = int(math.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for idx, metric_name in enumerate(metric_names):
        ax = flat_axes[idx]
        metric_df = df.filter(pl.col("metric") == metric_name)
        for mode in MODE_ORDER:
            mode_df = metric_df.filter(pl.col("mode") == mode).sort("date")
            if mode_df.is_empty():
                continue
            x = mode_df["date_dt"].to_list()
            y = mode_df["value"].to_list()
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
        _format_date_axis(ax)
        ax.legend(loc="best", fontsize=8)

    for idx in range(len(metric_names), len(flat_axes)):
        fig.delaxes(flat_axes[idx])

    fig.autofmt_xdate()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_cfg.dpi, bbox_inches="tight")
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def render_metrics_distribution(
    df: pl.DataFrame,
    metric_names: Sequence[str],
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
    ncols: int = 2,
) -> Optional[Path]:
    if not metric_names:
        return None
    ncols = max(1, min(ncols, len(metric_names)))
    nrows = int(math.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for idx, metric_name in enumerate(metric_names):
        ax = flat_axes[idx]
        metric_df = df.filter(pl.col("metric") == metric_name)
        data = []
        labels = []
        for mode in MODE_ORDER:
            mode_df = metric_df.filter(pl.col("mode") == mode)
            if mode_df.is_empty():
                continue
            data.append(mode_df["value"].to_numpy())
            labels.append(mode)
        if not data:
            ax.set_title(f"{metric_name} (no data)")
            ax.axis("off")
            continue
        ax.boxplot(data, tick_labels=labels)
        ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle="--")
        ax.set_title(metric_name)
        ax.set_ylabel("Daily value")
        ax.grid(alpha=0.2, axis="y")

    for idx in range(len(metric_names), len(flat_axes)):
        fig.delaxes(flat_axes[idx])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_cfg.dpi, bbox_inches="tight")
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def render_metrics_artifacts(
    df: Optional[pl.DataFrame],
    output_dir: Path,
    cfg: MetricsVizConfig,
    render_cfg: VisualizationRenderConfig,
) -> list[VisualizationArtifact]:
    if df is None or df.is_empty():
        return [
            _build_artifact(
                name="metrics",
                kind="metrics",
                status="skipped",
                message="No valid metrics data available.",
            )
        ]

    artifacts: list[VisualizationArtifact] = []
    try:
        selected = resolve_metric_names(df, cfg.metric_names)
    except Exception as exc:
        log.exception("Failed to resolve metric names for rendering.")
        return [_build_artifact(name="metrics", kind="metrics", status="failed", message=str(exc))]

    if not selected:
        return [
            _build_artifact(
                name="metrics",
                kind="metrics",
                status="skipped",
                message="No daily metrics available for rendering.",
            )
        ]

    if cfg.export_data:
        try:
            path = export_metrics_data_csv(df, output_dir / "metrics_data.csv", selected)
            if path is None:
                artifacts.append(
                    _build_artifact(
                        name="metrics_data",
                        kind="csv",
                        status="skipped",
                        message="No data to export.",
                    )
                )
            else:
                artifacts.append(_build_artifact(name="metrics_data", kind="csv", status="saved", path=path))
        except Exception as exc:
            log.exception("Failed to export metrics data CSV.")
            artifacts.append(_build_artifact(name="metrics_data", kind="csv", status="failed", message=str(exc)))

    if cfg.overview:
        try:
            path = render_metrics_overview(
                df=df,
                metric_names=selected,
                render_cfg=render_cfg,
                output_path=output_dir / "metrics_overview.png",
            )
            if path is None:
                artifacts.append(_build_artifact(name="metrics_overview", kind="image", status="skipped"))
            else:
                artifacts.append(_build_artifact(name="metrics_overview", kind="image", status="saved", path=path))
        except Exception as exc:
            log.exception("Failed to render metrics overview.")
            artifacts.append(_build_artifact(name="metrics_overview", kind="image", status="failed", message=str(exc)))

    if cfg.distribution:
        try:
            path = render_metrics_distribution(
                df=df,
                metric_names=selected,
                render_cfg=render_cfg,
                output_path=output_dir / "metrics_distribution.png",
            )
            if path is None:
                artifacts.append(_build_artifact(name="metrics_distribution", kind="image", status="skipped"))
            else:
                artifacts.append(
                    _build_artifact(name="metrics_distribution", kind="image", status="saved", path=path)
                )
        except Exception as exc:
            log.exception("Failed to render metrics distribution.")
            artifacts.append(
                _build_artifact(name="metrics_distribution", kind="image", status="failed", message=str(exc))
            )

    if cfg.per_metric:
        for metric_name in selected:
            safe_name = _safe_metric_filename(metric_name)
            output_path = output_dir / f"metric_{safe_name}_by_day.png"
            artifact_name = f"metric_{safe_name}_by_day"
            try:
                path = render_metric_by_day(df, metric_name, render_cfg, output_path)
                if path is None:
                    artifacts.append(_build_artifact(name=artifact_name, kind="image", status="skipped"))
                else:
                    artifacts.append(
                        _build_artifact(name=artifact_name, kind="image", status="saved", path=path)
                    )
            except Exception as exc:
                log.exception("Failed to render per-metric plot for %s.", metric_name)
                artifacts.append(_build_artifact(name=artifact_name, kind="image", status="failed", message=str(exc)))

    return artifacts
