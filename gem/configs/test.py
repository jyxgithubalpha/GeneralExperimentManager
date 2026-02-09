import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    data_module = instantiate(cfg.method.data.datamodule)
    print(data_module)

if __name__ == "__main__":
    main()
