"""
PyTorch Search Spaces - Architecture search spaces for neural networks.

Supports:
- MLP architecture search
- Transformer architecture search
- NNI integration
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..base.tuning.search_space import BaseSearchSpace


@dataclass
class MLPArchSpace(BaseSearchSpace):
    """
    MLP architecture search space.
    
    Searches over:
    - Number of layers
    - Hidden sizes per layer
    - Dropout rate
    - Activation function
    """
    
    n_layers: Tuple[int, int] = (2, 6)
    hidden_size: Tuple[int, int] = (64, 512)
    dropout: Tuple[float, float] = (0.0, 0.5)
    activation: List[str] = field(default_factory=lambda: ["relu", "gelu", "silu"])
    lr: Tuple[float, float] = (1e-4, 1e-2)
    weight_decay: Tuple[float, float] = (0.0, 0.1)
    batch_size: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        nl = space.get("n_layers", self.n_layers)
        hs = space.get("hidden_size", self.hidden_size)
        dp = space.get("dropout", self.dropout)
        act = space.get("activation", self.activation)
        lr = space.get("lr", self.lr)
        wd = space.get("weight_decay", self.weight_decay)
        bs = space.get("batch_size", self.batch_size)
        
        n_layers = trial.suggest_int("n_layers", int(nl[0]), int(nl[1]))
        hidden_sizes = [
            trial.suggest_int(f"hidden_{i}", int(hs[0]), int(hs[1]))
            for i in range(n_layers)
        ]
        
        return {
            "n_layers": n_layers,
            "hidden_sizes": hidden_sizes,
            "dropout": trial.suggest_float("dropout", *dp),
            "activation": trial.suggest_categorical("activation", act),
            "lr": trial.suggest_float("lr", *lr, log=True),
            "weight_decay": trial.suggest_float("weight_decay", *wd),
            "batch_size": trial.suggest_categorical("batch_size", bs),
        }
    
    def to_nni_space(self) -> Dict[str, Any]:
        """Convert to NNI search space format."""
        return {
            "n_layers": {"_type": "randint", "_value": [self.n_layers[0], self.n_layers[1] + 1]},
            "hidden_size": {"_type": "randint", "_value": [self.hidden_size[0], self.hidden_size[1] + 1]},
            "dropout": {"_type": "uniform", "_value": list(self.dropout)},
            "activation": {"_type": "choice", "_value": self.activation},
            "lr": {"_type": "loguniform", "_value": list(self.lr)},
            "weight_decay": {"_type": "uniform", "_value": list(self.weight_decay)},
            "batch_size": {"_type": "choice", "_value": self.batch_size},
        }


@dataclass
class TransformerArchSpace(BaseSearchSpace):
    """
    Transformer (FT-Transformer) architecture search space.
    
    Searches over:
    - Token dimension
    - Number of attention heads
    - Number of layers
    - Dropout rate
    """
    
    d_token: Tuple[int, int] = (32, 128)
    n_heads: List[int] = field(default_factory=lambda: [2, 4, 8])
    n_layers: Tuple[int, int] = (2, 6)
    dropout: Tuple[float, float] = (0.0, 0.3)
    lr: Tuple[float, float] = (1e-5, 1e-3)
    weight_decay: Tuple[float, float] = (0.0, 0.1)
    batch_size: List[int] = field(default_factory=lambda: [128, 256, 512])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        dt = space.get("d_token", self.d_token)
        nh = space.get("n_heads", self.n_heads)
        nl = space.get("n_layers", self.n_layers)
        dp = space.get("dropout", self.dropout)
        lr = space.get("lr", self.lr)
        wd = space.get("weight_decay", self.weight_decay)
        bs = space.get("batch_size", self.batch_size)
        
        d_token = trial.suggest_int("d_token", int(dt[0]), int(dt[1]))
        d_token = (d_token // 8) * 8
        if d_token < 8:
            d_token = 8
        
        return {
            "d_token": d_token,
            "n_heads": trial.suggest_categorical("n_heads", nh),
            "n_layers": trial.suggest_int("n_layers", int(nl[0]), int(nl[1])),
            "dropout": trial.suggest_float("dropout", *dp),
            "lr": trial.suggest_float("lr", *lr, log=True),
            "weight_decay": trial.suggest_float("weight_decay", *wd),
            "batch_size": trial.suggest_categorical("batch_size", bs),
        }
    
    def to_nni_space(self) -> Dict[str, Any]:
        """Convert to NNI search space format."""
        return {
            "d_token": {"_type": "choice", "_value": list(range(self.d_token[0], self.d_token[1] + 1, 8))},
            "n_heads": {"_type": "choice", "_value": self.n_heads},
            "n_layers": {"_type": "randint", "_value": [self.n_layers[0], self.n_layers[1] + 1]},
            "dropout": {"_type": "uniform", "_value": list(self.dropout)},
            "lr": {"_type": "loguniform", "_value": list(self.lr)},
            "weight_decay": {"_type": "uniform", "_value": list(self.weight_decay)},
            "batch_size": {"_type": "choice", "_value": self.batch_size},
        }


@dataclass
class TorchHyperSpace(BaseSearchSpace):
    """
    General PyTorch training hyperparameter space.
    
    For tuning training hyperparameters without architecture search.
    """
    
    lr: Tuple[float, float] = (1e-5, 1e-2)
    weight_decay: Tuple[float, float] = (0.0, 0.1)
    batch_size: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    epochs: Tuple[int, int] = (20, 100)
    patience: Tuple[int, int] = (5, 20)
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        lr = space.get("lr", self.lr)
        wd = space.get("weight_decay", self.weight_decay)
        bs = space.get("batch_size", self.batch_size)
        ep = space.get("epochs", self.epochs)
        pt = space.get("patience", self.patience)
        
        return {
            "lr": trial.suggest_float("lr", *lr, log=True),
            "weight_decay": trial.suggest_float("weight_decay", *wd),
            "batch_size": trial.suggest_categorical("batch_size", bs),
            "epochs": trial.suggest_int("epochs", int(ep[0]), int(ep[1])),
            "patience": trial.suggest_int("patience", int(pt[0]), int(pt[1])),
        }
    
    def to_nni_space(self) -> Dict[str, Any]:
        """Convert to NNI search space format."""
        return {
            "lr": {"_type": "loguniform", "_value": list(self.lr)},
            "weight_decay": {"_type": "uniform", "_value": list(self.weight_decay)},
            "batch_size": {"_type": "choice", "_value": self.batch_size},
            "epochs": {"_type": "randint", "_value": [self.epochs[0], self.epochs[1] + 1]},
            "patience": {"_type": "randint", "_value": [self.patience[0], self.patience[1] + 1]},
        }
