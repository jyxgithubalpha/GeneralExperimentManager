
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.animation import FuncAnimation

from ...experiment.configs import ImportanceVizConfig, VisualizationRenderConfig
from .contracts import VisualizationArtifact

log = logging.getLogger(__name__)


def _build_artifact(
    name: str,
    kind: str,
    status: str,
    path: Optional[Path] = None,
    message: Optional[str] = None,
) -> VisualizationArtifact:
    return VisualizationArtifact(name=name, kind=kind, status=status, path=path, message=message)


def _build_matrix(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_df = (
        df.select(["feature_idx", "feature_name"])
        .unique()
        .sort("feature_idx")
    )
    split_df = (
        df.select(["split_id", "x_label"])
        .unique()
        .sort("split_id")
    )
    n_features = feature_df.height
    if n_features == 0 or split_df.height == 0:
        raise ValueError("No valid importance data after preprocessing.")

    split_ids: list[int] = []
    labels: list[str] = []
    vectors: list[np.ndarray] = []

    for row in split_df.iter_rows(named=True):
        split_id = int(row["split_id"])
        split_rows = df.filter(pl.col("split_id") == split_id).sort("feature_idx")
        if split_rows.height != n_features:
            log.warning(
                "Skip split %s when rendering importance matrix: expected=%d got=%d",
                split_id,
                n_features,
                split_rows.height,
            )
            continue
        split_ids.append(split_id)
        labels.append(str(row["x_label"]))
        vectors.append(split_rows["importance"].to_numpy())

    if not vectors:
        raise ValueError("No split has complete importance vectors.")

    matrix = np.column_stack(vectors)
    return matrix, np.asarray(split_ids, dtype=np.int64), labels


def _resolve_sort_idx(values: np.ndarray, sort_by: str) -> np.ndarray:
    if sort_by == "mean":
        return np.argsort(np.mean(values, axis=1))[::-1]
    if sort_by == "std":
        return np.argsort(np.std(values, axis=1))[::-1]
    if sort_by == "max":
        return np.argsort(np.max(values, axis=1))[::-1]
    return np.arange(values.shape[0])


def _normalize(values: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "zscore":
        row_mean = np.mean(values, axis=1, keepdims=True)
        row_std = np.std(values, axis=1, keepdims=True) + 1e-10
        return (values - row_mean) / row_std
    if normalize == "minmax":
        row_min = np.min(values, axis=1, keepdims=True)
        row_max = np.max(values, axis=1, keepdims=True)
        return (values - row_min) / (row_max - row_min + 1e-10)
    if normalize == "rank":
        denom = max(values.shape[0] - 1, 1)
        return np.apply_along_axis(lambda x: np.argsort(np.argsort(x)) / denom, axis=0, arr=values)
    return values


def render_importance_csv(df: pl.DataFrame, output_path: Path) -> Optional[Path]:
    if df.is_empty():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    return output_path


def render_importance_heatmap(
    df: pl.DataFrame,
    cfg: ImportanceVizConfig,
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
) -> Optional[Path]:
    if df.is_empty():
        return None

    matrix, _, labels = _build_matrix(df)
    sort_idx = _resolve_sort_idx(matrix, cfg.sort_by)
    matrix_sorted = matrix[sort_idx]
    matrix_norm = _normalize(matrix_sorted, cfg.normalize)

    n_features, n_splits = matrix_norm.shape
    fig_height = max(10, min(100, n_features * 0.05))
    fig_width = max(12, min(30, n_splits * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(n_splits))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Split / Date")
    ax.set_ylabel(f"Feature (n={n_features})")
    ax.set_title(f"Feature Importance Heatmap (sorted by {cfg.sort_by}, {cfg.normalize})")
    plt.colorbar(im, ax=ax, label="Importance", shrink=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_cfg.dpi, bbox_inches="tight")
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def render_importance_animation(
    df: pl.DataFrame,
    cfg: ImportanceVizConfig,
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
) -> Optional[Path]:
    if df.is_empty():
        return None

    matrix, _, labels = _build_matrix(df)
    sort_idx = _resolve_sort_idx(matrix, cfg.sort_by)
    matrix_sorted = matrix[sort_idx]
    n_features, n_frames = matrix_sorted.shape

    max_val = float(np.max(matrix_sorted)) if matrix_sorted.size > 0 else 0.0
    max_val = max(max_val * 1.1, 1.0)

    fig_width = max(20, min(60, n_features * 0.02))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    x_pos = np.arange(n_features)
    bars = ax.bar(x_pos, np.zeros(n_features), width=0.8, color="steelblue", alpha=0.8)
    ax.set_ylim(0, max_val)
    ax.set_xlim(-0.5, n_features - 0.5)
    ax.set_ylabel("Importance")
    ax.set_xlabel(f"Feature (sorted by {cfg.sort_by}, n={n_features})")
    ax.set_xticks([])

    info_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    def init():
        for bar in bars:
            bar.set_height(0.0)
        info_text.set_text("")
        return [*bars, info_text]

    def update(frame_idx: int):
        values = matrix_sorted[:, frame_idx]
        for bar, val in zip(bars, values):
            bar.set_height(float(val))
        info_text.set_text(f"Frame: {labels[frame_idx]}")
        return [*bars, info_text]

    interval_ms = max(int(render_cfg.interval_ms), 1)
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=interval_ms,
        repeat=True,
        blit=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(int(1000 / interval_ms), 1)
    anim.save(str(output_path), writer="pillow", fps=fps)
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def render_importance_distribution(
    df: pl.DataFrame,
    render_cfg: VisualizationRenderConfig,
    output_path: Path,
) -> Optional[Path]:
    if df.is_empty():
        return None

    matrix, _, labels = _build_matrix(df)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([matrix[:, i] for i in range(matrix.shape[1])], tick_labels=labels)
    ax.set_xlabel("Split / Date")
    ax.set_ylabel("Importance")
    ax.set_title("Importance Distribution per Split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=render_cfg.dpi, bbox_inches="tight")
    if render_cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def render_importance_artifacts(
    df: Optional[pl.DataFrame],
    output_dir: Path,
    cfg: ImportanceVizConfig,
    render_cfg: VisualizationRenderConfig,
) -> list[VisualizationArtifact]:
    if df is None or df.is_empty():
        return [
            _build_artifact(
                name="importance",
                kind="importance",
                status="skipped",
                message="No valid importance data available.",
            )
        ]

    artifacts: list[VisualizationArtifact] = []
    jobs = []
    if cfg.export_data:
        jobs.append(("importance_data", "csv", output_dir / "importance_data.csv", render_importance_csv))
    if cfg.heatmap:
        jobs.append(("importance_heatmap", "image", output_dir / "importance_heatmap.png", render_importance_heatmap))
    if cfg.animation:
        jobs.append(("importance_animation", "animation", output_dir / "importance_animation.gif", render_importance_animation))
    if cfg.distribution:
        jobs.append(
            (
                "importance_distribution",
                "image",
                output_dir / "importance_distribution.png",
                render_importance_distribution,
            )
        )

    for name, kind, output_path, fn in jobs:
        try:
            if fn is render_importance_csv:
                path = fn(df, output_path)
            elif fn is render_importance_distribution:
                path = fn(df, render_cfg, output_path)
            else:
                path = fn(df, cfg, render_cfg, output_path)
            if path is None:
                artifacts.append(_build_artifact(name=name, kind=kind, status="skipped", message="No data to render."))
            else:
                artifacts.append(_build_artifact(name=name, kind=kind, status="saved", path=path))
        except Exception as exc:
            log.exception("Failed to render %s.", name)
            artifacts.append(_build_artifact(name=name, kind=kind, status="failed", message=str(exc)))
    return artifacts
