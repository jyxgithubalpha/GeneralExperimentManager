import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from .data import DataModule, SplitGenerator
from .experiment import ExperimentManager


log = logging.getLogger(__name__)


def _run_visualization_if_enabled(
    manager: ExperimentManager,
    results,
    output_dir: Path,
) -> None:
    viz_cfg = getattr(manager.experiment_config, "visualization", None)
    if viz_cfg is None or not bool(getattr(viz_cfg, "enabled", False)):
        return

    try:
        from .utils.importance_visualizer import ImportanceVisualizer
    except Exception as exc:
        log.warning(
            "Visualization is enabled but optional dependencies are unavailable: %s",
            exc,
        )
        return

    plot_subdir = str(getattr(viz_cfg, "output_subdir", "plots"))
    plot_dir = output_dir / plot_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    visualizer = ImportanceVisualizer(feature_names=manager.feature_names)
    visualizer.extract_from_results(results, manager.splitspec_list)

    if not visualizer.frames:
        log.warning("No valid importance data found, skip visualization.")
        return

    show = bool(getattr(viz_cfg, "show", False))
    interval = int(getattr(viz_cfg, "interval", 800))

    if bool(getattr(viz_cfg, "export_csv", True)):
        visualizer.export_to_csv(str(plot_dir / "importance_data.csv"))
    if bool(getattr(viz_cfg, "heatmap", True)):
        visualizer.plot_full_heatmap(
            output_path=str(plot_dir / "importance_heatmap.png"),
            show=show,
            sort_by="mean",
        )
    if bool(getattr(viz_cfg, "animation", True)):
        visualizer.create_importance_animation(
            output_path=str(plot_dir / "importance_animation.gif"),
            interval=interval,
            show=show,
            sort_by="mean",
        )
    if bool(getattr(viz_cfg, "distribution", True)):
        visualizer.plot_importance_distribution(
            output_path=str(plot_dir / "importance_distribution.png"),
            show=show,
        )


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    split_generator = instantiate(cfg.splitgenerator)
    data_module = instantiate(cfg.datamodule)
    experiment_config = instantiate(cfg.method.experiment_config)
    train_config = instantiate(cfg.method.train_config)
    
    # 从 method 配置获取 transform_pipeline 和 adapter
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
