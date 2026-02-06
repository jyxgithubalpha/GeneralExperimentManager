"""
Store - Artifact存储管理
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class Store:
    """Artifact存储管理"""
    
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
    
    def save_config(self, config: Dict[str, Any]) -> None:
        self._config_snapshot = config
        path = self.experiment_dir / "config.json"
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        path = self.experiment_dir / "results.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def load_importance(self, split_id: int) -> Optional[pd.DataFrame]:
        path = self.get_artifact_path(split_id, "feature_importance.csv")
        if path.exists():
            return pd.read_csv(path)
        return None
