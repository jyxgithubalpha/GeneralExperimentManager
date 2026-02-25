import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .bootstrap import build_runtime
from .experiment import ExperimentManager
from .utils import run_visualization_pipeline


log = logging.getLogger(__name__)


def _run_visualization_if_enabled(
    manager: ExperimentManager,
    results,
    output_dir: Path,
) -> None:
    viz_cfg = manager.experiment_config.visualization
    if not viz_cfg.enabled:
        return

    saved_paths = run_visualization_pipeline(
        manager=manager,
        results=results,
        output_dir=output_dir,
    )
    if not saved_paths:
        log.warning("Visualization enabled but no artifact was generated.")
        return
    for path in saved_paths:
        log.info("  - Saved: %s", path)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    runtime = build_runtime(cfg)

    manager = ExperimentManager(
        split_generator=runtime.split_generator,
        data_module=runtime.data_module,
        experiment_config=runtime.experiment_config,
        train_config=runtime.train_config,
        method_config=runtime.method_config,
    )

    log.info("Starting experiment: %s", runtime.experiment_config.name)
    results = manager.run()

    output_dir = Path(runtime.experiment_config.output_dir)
    _run_visualization_if_enabled(manager, results, output_dir)

    return results


if __name__ == "__main__":
    main()
