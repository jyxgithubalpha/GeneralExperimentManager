import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from .data import DataModule, SplitGenerator
from .experiment import ExperimentManager


log = logging.getLogger(__name__)


def _as_str_list(value) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return None


def _run_importance_visualization(
    manager: ExperimentManager,
    results,
    plot_dir: Path,
    show: bool,
    interval: int,
    viz_cfg,
) -> None:
    if not any(
        bool(getattr(viz_cfg, name, False))
        for name in ("export_csv", "heatmap", "animation", "distribution")
    ):
        return

    try:
        from .utils.importance_visualizer import ImportanceVisualizer
    except Exception as exc:
        log.warning(
            "Skipping importance visualization because dependencies are unavailable: %s",
            exc,
        )
        return

    visualizer = ImportanceVisualizer(feature_names=manager.feature_names)
    visualizer.extract_from_results(results, manager.splitspec_list)

    if not visualizer.frames:
        log.warning("No valid feature-importance data found, skip importance plots.")
        return

    if bool(getattr(viz_cfg, "export_csv", True)):
        path = plot_dir / "importance_data.csv"
        visualizer.export_to_csv(str(path))
        log.info("  - Saved: %s", path)
    if bool(getattr(viz_cfg, "heatmap", True)):
        path = plot_dir / "importance_heatmap.png"
        visualizer.plot_full_heatmap(
            output_path=str(path),
            show=show,
            sort_by="mean",
        )
        log.info("  - Saved: %s", path)
    if bool(getattr(viz_cfg, "animation", True)):
        path = plot_dir / "importance_animation.gif"
        visualizer.create_importance_animation(
            output_path=str(path),
            interval=interval,
            show=show,
            sort_by="mean",
        )
        log.info("  - Saved: %s", path)
    if bool(getattr(viz_cfg, "distribution", True)):
        path = plot_dir / "importance_distribution.png"
        visualizer.plot_importance_distribution(
            output_path=str(path),
            show=show,
        )
        log.info("  - Saved: %s", path)


def _run_metrics_visualization(
    output_dir: Path,
    plot_dir: Path,
    show: bool,
    viz_cfg,
) -> None:
    if not bool(getattr(viz_cfg, "metrics", True)):
        return

    series_path = output_dir / "daily_metric_series.csv"
    if not series_path.exists():
        log.warning("daily_metric_series.csv not found, skip metric visualization.")
        return

    try:
        from .utils.metric_visualizer import MetricsVisualizer
    except Exception as exc:
        log.warning(
            "Skipping metric visualization because dependencies are unavailable: %s",
            exc,
        )
        return

    visualizer = MetricsVisualizer()
    metric_df = visualizer.load_from_daily_series_csv(series_path)
    if metric_df is None or metric_df.is_empty():
        log.warning("No valid daily metrics found, skip metric plots.")
        return

    metric_names = _as_str_list(getattr(viz_cfg, "metric_names", None))
    saved_paths = visualizer.create_all_plots(
        output_dir=plot_dir,
        metric_names=metric_names,
        show=show,
        export_csv=bool(getattr(viz_cfg, "metrics_export_csv", True)),
        overview=bool(getattr(viz_cfg, "metrics_overview", True)),
        distribution=bool(getattr(viz_cfg, "metrics_distribution", True)),
        per_metric=bool(getattr(viz_cfg, "metrics_per_metric", True)),
    )
    if not saved_paths:
        log.warning("Metric visualization enabled but no metric plot was generated.")
        return
    for path in saved_paths:
        log.info("  - Saved: %s", path)


def _run_visualization_if_enabled(
    manager: ExperimentManager,
    results,
    output_dir: Path,
) -> None:
    viz_cfg = getattr(manager.experiment_config, "visualization", None)
    if viz_cfg is None or not bool(getattr(viz_cfg, "enabled", False)):
        return

    plot_subdir = str(getattr(viz_cfg, "output_subdir", "plots"))
    plot_dir = output_dir / plot_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    show = bool(getattr(viz_cfg, "show", False))
    interval = int(getattr(viz_cfg, "interval", 800))

    _run_importance_visualization(
        manager=manager,
        results=results,
        plot_dir=plot_dir,
        show=show,
        interval=interval,
        viz_cfg=viz_cfg,
    )
    _run_metrics_visualization(
        output_dir=output_dir,
        plot_dir=plot_dir,
        show=show,
        viz_cfg=viz_cfg,
    )


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    split_generator = instantiate(cfg.splitgenerator)
    data_module = instantiate(cfg.datamodule)
    experiment_config = instantiate(cfg.method.experiment_config)
    train_config = instantiate(cfg.method.train_config)
    transform_pipeline_config = cfg.method.get("transform_pipeline", None)
    adapter_config = cfg.method.get("adapter", None)
    method_config = cfg.method
    
    manager = ExperimentManager(
        split_generator=split_generator,
        data_module=data_module,
        experiment_config=experiment_config,
        train_config=train_config,
        method_config=method_config,
        transform_pipeline_config=transform_pipeline_config,
        adapter_config=adapter_config,
    )
    
    log.info("Starting experiment: %s", experiment_config.name)
    results = manager.run()

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    _run_visualization_if_enabled(manager, results, output_dir)
    
    return results


if __name__ == "__main__":
    main()
