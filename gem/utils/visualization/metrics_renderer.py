
import logging
import math
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from ...experiment.configs import MetricsVizConfig, VisualizationRenderConfig
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


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    get_fn = getattr(cfg, "get", None)
    if callable(get_fn):
        try:
            value = get_fn(key, default)
            return default if value is None else value
        except Exception:
            pass
    try:
        value = getattr(cfg, key)
        return default if value is None else value
    except Exception:
        return default


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


def _build_backtest_curve_frame(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    model_df = (
        df.filter(pl.col("metric") == "daily_top_ret")
        .select(["date", "date_dt", "mode", "value"])
        .rename({"value": "model_ret"})
    )
    bench_df = (
        df.filter(pl.col("metric") == "daily_benchmark_top_ret")
        .select(["date", "date_dt", "mode", "value"])
        .rename({"value": "bench_ret"})
    )
    if model_df.is_empty() or bench_df.is_empty():
        return None

    joined = model_df.join(bench_df, on=["date", "date_dt", "mode"], how="inner").sort(["mode", "date"])
    if joined.is_empty():
        return None

    return joined.with_columns(
        pl.col("model_ret").cum_sum().over("mode").alias("model_cum_ret"),
        pl.col("bench_ret").cum_sum().over("mode").alias("bench_cum_ret"),
        (pl.col("model_ret") - pl.col("bench_ret")).alias("daily_excess_ret"),
    ).with_columns(
        pl.col("daily_excess_ret").cum_sum().over("mode").alias("excess_cum_ret"),
    )


def _quarterly_summary(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    quarter_expr = (
        pl.col("date_dt").dt.year().cast(pl.Utf8)
        + pl.lit("Q")
        + pl.col("date_dt").dt.quarter().cast(pl.Utf8)
    ).alias("quarter")

    top_quarterly = (
        df.filter(pl.col("metric") == "daily_top_ret")
        .with_columns(quarter_expr)
        .group_by(["mode", "quarter"])
        .agg(pl.col("value").mean().alias("quarterly_mean_top_ret"))
        .sort(["mode", "quarter"])
    )
    ic_quarterly = (
        df.filter(pl.col("metric") == "daily_ic")
        .with_columns(quarter_expr)
        .group_by(["mode", "quarter"])
        .agg(pl.col("value").mean().alias("quarterly_mean_ic"))
        .sort(["mode", "quarter"])
    )

    if top_quarterly.is_empty() and ic_quarterly.is_empty():
        return None

    frames: list[pl.DataFrame] = []
    if not top_quarterly.is_empty():
        frames.append(
            top_quarterly.with_columns(pl.lit(None, dtype=pl.Float64).alias("quarterly_mean_ic")).select(
                ["mode", "quarter", "quarterly_mean_top_ret", "quarterly_mean_ic"]
            )
        )
    if not ic_quarterly.is_empty():
        frames.append(
            ic_quarterly.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("quarterly_mean_top_ret")
            ).select(
                ["mode", "quarter", "quarterly_mean_top_ret", "quarterly_mean_ic"]
            )
        )

    combined = pl.concat(frames, how="vertical")
    return combined.group_by(["mode", "quarter"]).agg(
        pl.col("quarterly_mean_top_ret").drop_nulls().first().alias("quarterly_mean_top_ret"),
        pl.col("quarterly_mean_ic").drop_nulls().first().alias("quarterly_mean_ic"),
    ).sort(["mode", "quarter"])


def export_quarterly_summary_csv(df: pl.DataFrame, output_path: Path) -> Optional[Path]:
    summary = _quarterly_summary(df)
    if summary is None or summary.is_empty():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(output_path)
    return output_path


def render_cumulative_return_compare(
    df: pl.DataFrame,
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
) -> Optional[Path]:
    curve_df = _build_backtest_curve_frame(df)
    if curve_df is None or curve_df.is_empty():
        return None

    mode_values = curve_df["mode"].unique().to_list()
    modes = [mode for mode in MODE_ORDER if mode in mode_values] + [
        mode for mode in mode_values if mode not in MODE_ORDER
    ]
    if not modes:
        return None

    nrows = len(modes)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, max(4, 3.5 * nrows)), squeeze=False)

    for idx, mode in enumerate(modes):
        ax = axes[idx][0]
        mode_df = curve_df.filter(pl.col("mode") == mode).sort("date")
        if mode_df.is_empty():
            continue
        x = mode_df["date_dt"].to_list()
        ax.plot(x, mode_df["model_cum_ret"].to_list(), label="model", linewidth=1.8, color="#1f77b4")
        ax.plot(x, mode_df["bench_cum_ret"].to_list(), label="benchmark", linewidth=1.8, color="#ff7f0e")
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_title(f"{mode} cumulative return")
        ax.set_ylabel("Cumulative Return")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        _format_date_axis(ax)

    axes[-1][0].set_xlabel("Date")
    fig.suptitle("Cumulative Return Of Model & Benchmark", fontsize=13)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_cfg.dpi, bbox_inches="tight")
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def render_relative_improvement_curve(
    df: pl.DataFrame,
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
) -> Optional[Path]:
    curve_df = _build_backtest_curve_frame(df)
    if curve_df is None or curve_df.is_empty():
        return None

    mode_values = curve_df["mode"].unique().to_list()
    modes = [mode for mode in MODE_ORDER if mode in mode_values] + [
        mode for mode in mode_values if mode not in MODE_ORDER
    ]
    if not modes:
        return None

    nrows = len(modes)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, max(4, 3.5 * nrows)), squeeze=False)

    for idx, mode in enumerate(modes):
        ax = axes[idx][0]
        mode_df = curve_df.filter(pl.col("mode") == mode).sort("date")
        if mode_df.is_empty():
            continue
        x = mode_df["date_dt"].to_list()
        ax.plot(x, mode_df["excess_cum_ret"].to_list(), label="model - benchmark", linewidth=1.8, color="#1f77b4")
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_title(f"{mode} relative improvement (cumulative)")
        ax.set_ylabel("Cumulative Excess Return")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        _format_date_axis(ax)

    axes[-1][0].set_xlabel("Date")
    fig.suptitle("Relative Improvement", fontsize=13)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_cfg.dpi, bbox_inches="tight")
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
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
    metric_names_cfg = _cfg_get(cfg, "metric_names", None)
    export_data = bool(_cfg_get(cfg, "export_data", True))
    quarterly_summary = bool(_cfg_get(cfg, "quarterly_summary", True))
    cumulative_backtest = bool(_cfg_get(cfg, "cumulative_backtest", True))
    relative_improvement_curve = bool(_cfg_get(cfg, "relative_improvement_curve", True))
    overview = bool(_cfg_get(cfg, "overview", True))
    distribution = bool(_cfg_get(cfg, "distribution", True))
    per_metric = bool(_cfg_get(cfg, "per_metric", True))
    try:
        selected = resolve_metric_names(df, metric_names_cfg)
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

    if export_data:
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

    if quarterly_summary:
        try:
            path = export_quarterly_summary_csv(df, output_dir / "metrics_quarterly_summary.csv")
            if path is None:
                artifacts.append(
                    _build_artifact(
                        name="metrics_quarterly_summary",
                        kind="csv",
                        status="skipped",
                        message="No daily_top_ret/daily_ic data for quarterly summary.",
                    )
                )
            else:
                artifacts.append(
                    _build_artifact(
                        name="metrics_quarterly_summary",
                        kind="csv",
                        status="saved",
                        path=path,
                    )
                )
        except Exception as exc:
            log.exception("Failed to export quarterly summary CSV.")
            artifacts.append(
                _build_artifact(
                    name="metrics_quarterly_summary",
                    kind="csv",
                    status="failed",
                    message=str(exc),
                )
            )

    if cumulative_backtest:
        try:
            path = render_cumulative_return_compare(
                df=df,
                render_cfg=render_cfg,
                output_path=output_dir / "metrics_cumulative_return_compare.png",
            )
            if path is None:
                artifacts.append(
                    _build_artifact(
                        name="metrics_cumulative_return_compare",
                        kind="image",
                        status="skipped",
                        message="No paired daily_top_ret and daily_benchmark_top_ret data.",
                    )
                )
            else:
                artifacts.append(
                    _build_artifact(
                        name="metrics_cumulative_return_compare",
                        kind="image",
                        status="saved",
                        path=path,
                    )
                )
        except Exception as exc:
            log.exception("Failed to render cumulative model/benchmark return comparison.")
            artifacts.append(
                _build_artifact(
                    name="metrics_cumulative_return_compare",
                    kind="image",
                    status="failed",
                    message=str(exc),
                )
            )

    if relative_improvement_curve:
        try:
            path = render_relative_improvement_curve(
                df=df,
                render_cfg=render_cfg,
                output_path=output_dir / "metrics_relative_improvement_curve.png",
            )
            if path is None:
                artifacts.append(
                    _build_artifact(
                        name="metrics_relative_improvement_curve",
                        kind="image",
                        status="skipped",
                        message="No paired daily_top_ret and daily_benchmark_top_ret data.",
                    )
                )
            else:
                artifacts.append(
                    _build_artifact(
                        name="metrics_relative_improvement_curve",
                        kind="image",
                        status="saved",
                        path=path,
                    )
                )
        except Exception as exc:
            log.exception("Failed to render relative improvement curve.")
            artifacts.append(
                _build_artifact(
                    name="metrics_relative_improvement_curve",
                    kind="image",
                    status="failed",
                    message=str(exc),
                )
            )

    if overview:
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

    if distribution:
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

    if per_metric:
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
