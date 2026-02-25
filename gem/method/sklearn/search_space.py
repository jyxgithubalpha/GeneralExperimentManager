"""
Sklearn Search Spaces - Parameter spaces for sklearn models.

Supports:
- Random Forest
- Gradient Boosting
- Linear models (Ridge, Lasso, ElasticNet)
- SVR
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..base.tuning.search_space import BaseSearchSpace


@dataclass
class RandomForestSpace(BaseSearchSpace):
    """Random Forest Regressor hyperparameter search space."""
    
    n_estimators: Tuple[int, int] = (50, 500)
    max_depth: Tuple[int, int] = (3, 20)
    min_samples_split: Tuple[int, int] = (2, 20)
    min_samples_leaf: Tuple[int, int] = (1, 10)
    max_features: List[str] = field(default_factory=lambda: ["sqrt", "log2", None])
    bootstrap: List[bool] = field(default_factory=lambda: [True, False])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        ne = space.get("n_estimators", self.n_estimators)
        md = space.get("max_depth", self.max_depth)
        mss = space.get("min_samples_split", self.min_samples_split)
        msl = space.get("min_samples_leaf", self.min_samples_leaf)
        mf = space.get("max_features", self.max_features)
        bs = space.get("bootstrap", self.bootstrap)
        
        return {
            "n_estimators": trial.suggest_int("n_estimators", int(ne[0]), int(ne[1])),
            "max_depth": trial.suggest_int("max_depth", int(md[0]), int(md[1])),
            "min_samples_split": trial.suggest_int("min_samples_split", int(mss[0]), int(mss[1])),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", int(msl[0]), int(msl[1])),
            "max_features": trial.suggest_categorical("max_features", mf),
            "bootstrap": trial.suggest_categorical("bootstrap", bs),
        }
    
    def to_ray_tune_space(self, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from ray import tune
        except ImportError:
            raise ImportError("ray[tune] is required. Install with: pip install 'ray[tune]'")
        
        space = shrunk_space or {}
        
        ne = space.get("n_estimators", self.n_estimators)
        md = space.get("max_depth", self.max_depth)
        mss = space.get("min_samples_split", self.min_samples_split)
        msl = space.get("min_samples_leaf", self.min_samples_leaf)
        mf = space.get("max_features", self.max_features)
        bs = space.get("bootstrap", self.bootstrap)
        
        return {
            "n_estimators": tune.randint(int(ne[0]), int(ne[1]) + 1),
            "max_depth": tune.randint(int(md[0]), int(md[1]) + 1),
            "min_samples_split": tune.randint(int(mss[0]), int(mss[1]) + 1),
            "min_samples_leaf": tune.randint(int(msl[0]), int(msl[1]) + 1),
            "max_features": tune.choice(mf),
            "bootstrap": tune.choice(bs),
        }


@dataclass
class GradientBoostingSpace(BaseSearchSpace):
    """Gradient Boosting Regressor hyperparameter search space."""
    
    n_estimators: Tuple[int, int] = (50, 500)
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    max_depth: Tuple[int, int] = (3, 10)
    min_samples_split: Tuple[int, int] = (2, 20)
    min_samples_leaf: Tuple[int, int] = (1, 10)
    subsample: Tuple[float, float] = (0.5, 1.0)
    max_features: List[str] = field(default_factory=lambda: ["sqrt", "log2", None])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        ne = space.get("n_estimators", self.n_estimators)
        lr = space.get("learning_rate", self.learning_rate)
        md = space.get("max_depth", self.max_depth)
        mss = space.get("min_samples_split", self.min_samples_split)
        msl = space.get("min_samples_leaf", self.min_samples_leaf)
        ss = space.get("subsample", self.subsample)
        mf = space.get("max_features", self.max_features)
        
        return {
            "n_estimators": trial.suggest_int("n_estimators", int(ne[0]), int(ne[1])),
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "max_depth": trial.suggest_int("max_depth", int(md[0]), int(md[1])),
            "min_samples_split": trial.suggest_int("min_samples_split", int(mss[0]), int(mss[1])),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", int(msl[0]), int(msl[1])),
            "subsample": trial.suggest_float("subsample", *ss),
            "max_features": trial.suggest_categorical("max_features", mf),
        }


@dataclass
class RidgeSpace(BaseSearchSpace):
    """Ridge Regression hyperparameter search space."""
    
    alpha: Tuple[float, float] = (1e-4, 100.0)
    solver: List[str] = field(default_factory=lambda: ["auto", "svd", "cholesky", "lsqr", "sag"])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        alpha = space.get("alpha", self.alpha)
        solver = space.get("solver", self.solver)
        
        return {
            "alpha": trial.suggest_float("alpha", *alpha, log=True),
            "solver": trial.suggest_categorical("solver", solver),
        }


@dataclass
class LassoSpace(BaseSearchSpace):
    """Lasso Regression hyperparameter search space."""
    
    alpha: Tuple[float, float] = (1e-4, 100.0)
    selection: List[str] = field(default_factory=lambda: ["cyclic", "random"])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        alpha = space.get("alpha", self.alpha)
        selection = space.get("selection", self.selection)
        
        return {
            "alpha": trial.suggest_float("alpha", *alpha, log=True),
            "selection": trial.suggest_categorical("selection", selection),
        }


@dataclass
class ElasticNetSpace(BaseSearchSpace):
    """ElasticNet Regression hyperparameter search space."""
    
    alpha: Tuple[float, float] = (1e-4, 100.0)
    l1_ratio: Tuple[float, float] = (0.0, 1.0)
    selection: List[str] = field(default_factory=lambda: ["cyclic", "random"])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        alpha = space.get("alpha", self.alpha)
        l1_ratio = space.get("l1_ratio", self.l1_ratio)
        selection = space.get("selection", self.selection)
        
        return {
            "alpha": trial.suggest_float("alpha", *alpha, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", *l1_ratio),
            "selection": trial.suggest_categorical("selection", selection),
        }


@dataclass
class SVRSpace(BaseSearchSpace):
    """SVR (Support Vector Regression) hyperparameter search space."""
    
    C: Tuple[float, float] = (1e-2, 100.0)
    epsilon: Tuple[float, float] = (0.01, 1.0)
    kernel: List[str] = field(default_factory=lambda: ["rbf", "linear", "poly"])
    gamma: List[str] = field(default_factory=lambda: ["scale", "auto"])
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        C = space.get("C", self.C)
        epsilon = space.get("epsilon", self.epsilon)
        kernel = space.get("kernel", self.kernel)
        gamma = space.get("gamma", self.gamma)
        
        return {
            "C": trial.suggest_float("C", *C, log=True),
            "epsilon": trial.suggest_float("epsilon", *epsilon),
            "kernel": trial.suggest_categorical("kernel", kernel),
            "gamma": trial.suggest_categorical("gamma", gamma),
        }


@dataclass
class HistGradientBoostingSpace(BaseSearchSpace):
    """HistGradientBoosting Regressor hyperparameter search space."""
    
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    max_iter: Tuple[int, int] = (100, 1000)
    max_depth: Tuple[int, int] = (3, 15)
    min_samples_leaf: Tuple[int, int] = (10, 100)
    l2_regularization: Tuple[float, float] = (0.0, 10.0)
    max_bins: Tuple[int, int] = (128, 255)
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        lr = space.get("learning_rate", self.learning_rate)
        mi = space.get("max_iter", self.max_iter)
        md = space.get("max_depth", self.max_depth)
        msl = space.get("min_samples_leaf", self.min_samples_leaf)
        l2 = space.get("l2_regularization", self.l2_regularization)
        mb = space.get("max_bins", self.max_bins)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "max_iter": trial.suggest_int("max_iter", int(mi[0]), int(mi[1])),
            "max_depth": trial.suggest_int("max_depth", int(md[0]), int(md[1])),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", int(msl[0]), int(msl[1])),
            "l2_regularization": trial.suggest_float("l2_regularization", *l2),
            "max_bins": trial.suggest_int("max_bins", int(mb[0]), int(mb[1])),
        }
