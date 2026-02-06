import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from .data import DataModule, SplitGenerator
from .experiment import ExperimentManager
from .utils.importance_visualizer import visualize_importance_animation


log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    split_generator = instantiate(cfg.splitgenerator)
    data_module = instantiate(cfg.datamodule)
    experiment_config = instantiate(cfg.method.experiment_config)
    train_config = instantiate(cfg.method.train_config)
    
    data_processor_config = cfg.get("data_processor", None)
    method_config = cfg.method
    
    manager = ExperimentManager(
        split_generator=split_generator,
        data_module=data_module,
        experiment_config=experiment_config,
        train_config=train_config,
        method_config=method_config,
        data_processor_config=data_processor_config,
    )
    
    log.info("Starting experiment: %s", experiment_config.name)
    results = manager.run()

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    visualize_importance_animation(
        results=results,
        feature_names=manager.feature_names,
        splitspec_list=manager.splitspec_list,
        output_dir=str(output_dir),
        interval=800,
        show=False,
    )
    
    return results


if __name__ == "__main__":
    main()
