"""
Store - Artifact瀛樺偍绠＄悊
"""



import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl


class Store:
    """Artifact瀛樺偍绠＄悊"""
    
    def __init__(self, base_dir: Path, experiment_name: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / self.experiment_name
        self._init_dirs()
        self._config_snapshot: Dict[str, Any] = {}
    
    def _init_dirs(self) -> None:
        (self.experiment_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, split_id: int) -> Path:
        return self.experiment_dir / "models" / f"model_split_{split_id}.txt"
    
    def get_artifact_path(self, split_id: int, filename: str) -> Path:
        split_dir = self.experiment_dir / "artifacts" / f"split_{split_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir / filename
    
    def get_plot_path(self, filename: str) -> Path:
        return self.experiment_dir / "plots" / filename
    
    def split_dir(self, split_id: int) -> Path:
        """鑾峰彇 split 杈撳嚭鐩綍"""
        d = self.experiment_dir / f"split_{split_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def trial_dir(self, split_id: int, trial_id: int) -> Path:
        """鑾峰彇 trial 杈撳嚭鐩綍"""
        d = self.split_dir(split_id) / f"trial_{trial_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def save_artifact(self, split_id: int, name: str, obj: Any) -> Path:
        """淇濆瓨 artifact"""
        import pickle
        path = self.split_dir(split_id) / name
        suffix = path.suffix.lower()

        if isinstance(obj, pl.DataFrame):
            if suffix == ".parquet":
                obj.write_parquet(str(path))
            else:
                if suffix not in {"", ".csv"}:
                    raise ValueError(
                        f"Unsupported DataFrame artifact suffix '{suffix}'. "
                        "Use .csv or .parquet."
                    )
                if suffix == "":
                    path = path.with_suffix(".csv")
                obj.write_csv(str(path))
        else:
            if suffix == "":
                path = path.with_suffix(".pkl")
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        return path
    
    def save_config(self, config: Dict[str, Any]) -> None:
        self._config_snapshot = config
        path = self.experiment_dir / "config.json"
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        path = self.experiment_dir / "results.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def load_importance(self, split_id: int) -> Optional[pl.DataFrame]:
        path = self.get_artifact_path(split_id, "feature_importance.csv")
        if path.exists():
            return pl.read_csv(path)
        return None

