from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import VisualizationArtifact
from .importance_data import build_importance_dataframe
from .importance_renderer import render_importance_artifacts
from .manifest import write_manifest
from .metrics_data import load_metrics_data
from .metrics_renderer import render_metrics_artifacts

if TYPE_CHECKING:
    from ..experiment import ExperimentManager
    from ..experiment.experiment_dataclasses import SplitResult

log = logging.getLogger(__name__)


def run_visualization_pipeline(
    manager: "ExperimentManager",
    results: dict[int, "SplitResult"],
    output_dir: Path,
) -> list[Path]:
    viz_cfg = manager.experiment_config.visualization
    plot_dir = output_dir / viz_cfg.output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[VisualizationArtifact] = []

    if viz_cfg.importance.enabled:
        importance_df = build_importance_dataframe(
            results=results,
            splitspec_list=manager.splitspec_list,
            feature_names=manager.feature_names,
        )
        artifacts.extend(
            render_importance_artifacts(
                df=importance_df,
                output_dir=plot_dir,
                cfg=viz_cfg.importance,
                render_cfg=viz_cfg.render,
            )
        )
    else:
        artifacts.append(
            VisualizationArtifact(
                name="importance",
                kind="importance",
                status="skipped",
                message="Importance rendering is disabled by config.",
            )
        )

    if viz_cfg.metrics.enabled:
        metric_df, err = load_metrics_data(output_dir / "daily_metric_series.csv")
        if metric_df is None:
            artifacts.append(
                VisualizationArtifact(
                    name="metrics",
                    kind="metrics",
                    status="skipped",
                    message=err,
                )
            )
        else:
            artifacts.extend(
                render_metrics_artifacts(
                    df=metric_df,
                    output_dir=plot_dir,
                    cfg=viz_cfg.metrics,
                    render_cfg=viz_cfg.render,
                )
            )
    else:
        artifacts.append(
            VisualizationArtifact(
                name="metrics",
                kind="metrics",
                status="skipped",
                message="Metric rendering is disabled by config.",
            )
        )

    manifest_path = write_manifest(plot_dir / "manifest.json", artifacts)
    saved_paths = [
        item.path
        for item in artifacts
        if item.status == "saved" and item.path is not None
    ]
    saved_paths.append(manifest_path)

    n_failed = sum(1 for item in artifacts if item.status == "failed")
    if n_failed > 0:
        log.warning("Visualization pipeline finished with %d failed artifact(s).", n_failed)

    return saved_paths

