import hydra
from omegaconf import DictConfig
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gem.bootstrap import build_runtime


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    runtime = build_runtime(cfg)
    print(runtime.split_generator)
    print(runtime.data_module)
    print(runtime.experiment_config)
    print(runtime.train_config)
    print("method name:", cfg.method.name)

if __name__ == "__main__":
    main()
