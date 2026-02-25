"""
GBDT Search Spaces - Parameter spaces for LightGBM, XGBoost, CatBoost.
"""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..base.tuning.search_space import BaseSearchSpace


@dataclass
class LightGBMSpace(BaseSearchSpace):
    """LightGBM hyperparameter search space."""
    
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    num_leaves: Tuple[int, int] = (20, 300)
    max_depth: Tuple[int, int] = (3, 12)
    min_child_samples: Tuple[int, int] = (5, 100)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    reg_alpha: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda: Tuple[float, float] = (1e-8, 10.0)
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        lr = space.get("learning_rate", self.learning_rate)
        nl = space.get("num_leaves", self.num_leaves)
        md = space.get("max_depth", self.max_depth)
        mcs = space.get("min_child_samples", self.min_child_samples)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "num_leaves": trial.suggest_int("num_leaves", int(nl[0]), int(nl[1])),
            "max_depth": trial.suggest_int("max_depth", int(md[0]), int(md[1])),
            "min_child_samples": trial.suggest_int("min_child_samples", int(mcs[0]), int(mcs[1])),
            "subsample": trial.suggest_float("subsample", *ss),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cs),
            "reg_alpha": trial.suggest_float("reg_alpha", *ra, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *rl, log=True),
        }
    
    def to_ray_tune_space(self, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from ray import tune
        except ImportError:
            raise ImportError("ray[tune] is required. Install with: pip install 'ray[tune]'")
        
        space = shrunk_space or {}
        
        lr = space.get("learning_rate", self.learning_rate)
        nl = space.get("num_leaves", self.num_leaves)
        md = space.get("max_depth", self.max_depth)
        mcs = space.get("min_child_samples", self.min_child_samples)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)
        
        return {
            "learning_rate": tune.loguniform(*lr),
            "num_leaves": tune.randint(int(nl[0]), int(nl[1]) + 1),
            "max_depth": tune.randint(int(md[0]), int(md[1]) + 1),
            "min_child_samples": tune.randint(int(mcs[0]), int(mcs[1]) + 1),
            "subsample": tune.uniform(*ss),
            "colsample_bytree": tune.uniform(*cs),
            "reg_alpha": tune.loguniform(*ra),
            "reg_lambda": tune.loguniform(*rl),
        }


@dataclass
class XGBoostSpace(BaseSearchSpace):
    """XGBoost hyperparameter search space."""
    
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    max_depth: Tuple[int, int] = (3, 12)
    min_child_weight: Tuple[int, int] = (1, 10)
    subsample: Tuple[float, float] = (0.5, 1.0)
    colsample_bytree: Tuple[float, float] = (0.5, 1.0)
    reg_alpha: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda: Tuple[float, float] = (1e-8, 10.0)
    gamma: Tuple[float, float] = (0.0, 5.0)
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        lr = space.get("learning_rate", self.learning_rate)
        md = space.get("max_depth", self.max_depth)
        mcw = space.get("min_child_weight", self.min_child_weight)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)
        gm = space.get("gamma", self.gamma)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "max_depth": trial.suggest_int("max_depth", int(md[0]), int(md[1])),
            "min_child_weight": trial.suggest_int("min_child_weight", int(mcw[0]), int(mcw[1])),
            "subsample": trial.suggest_float("subsample", *ss),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cs),
            "reg_alpha": trial.suggest_float("reg_alpha", *ra, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *rl, log=True),
            "gamma": trial.suggest_float("gamma", *gm),
        }
    
    def to_ray_tune_space(self, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from ray import tune
        except ImportError:
            raise ImportError("ray[tune] is required. Install with: pip install 'ray[tune]'")
        
        space = shrunk_space or {}
        
        lr = space.get("learning_rate", self.learning_rate)
        md = space.get("max_depth", self.max_depth)
        mcw = space.get("min_child_weight", self.min_child_weight)
        ss = space.get("subsample", self.subsample)
        cs = space.get("colsample_bytree", self.colsample_bytree)
        ra = space.get("reg_alpha", self.reg_alpha)
        rl = space.get("reg_lambda", self.reg_lambda)
        gm = space.get("gamma", self.gamma)
        
        return {
            "learning_rate": tune.loguniform(*lr),
            "max_depth": tune.randint(int(md[0]), int(md[1]) + 1),
            "min_child_weight": tune.randint(int(mcw[0]), int(mcw[1]) + 1),
            "subsample": tune.uniform(*ss),
            "colsample_bytree": tune.uniform(*cs),
            "reg_alpha": tune.loguniform(*ra),
            "reg_lambda": tune.loguniform(*rl),
            "gamma": tune.uniform(*gm),
        }


@dataclass
class CatBoostSpace(BaseSearchSpace):
    """CatBoost hyperparameter search space."""
    
    learning_rate: Tuple[float, float] = (0.01, 0.3)
    depth: Tuple[int, int] = (4, 10)
    l2_leaf_reg: Tuple[float, float] = (1.0, 10.0)
    bagging_temperature: Tuple[float, float] = (0.0, 1.0)
    random_strength: Tuple[float, float] = (0.0, 10.0)
    border_count: Tuple[int, int] = (32, 255)
    
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        space = shrunk_space or {}
        
        lr = space.get("learning_rate", self.learning_rate)
        dp = space.get("depth", self.depth)
        l2 = space.get("l2_leaf_reg", self.l2_leaf_reg)
        bt = space.get("bagging_temperature", self.bagging_temperature)
        rs = space.get("random_strength", self.random_strength)
        bc = space.get("border_count", self.border_count)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
            "depth": trial.suggest_int("depth", int(dp[0]), int(dp[1])),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *l2),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *bt),
            "random_strength": trial.suggest_float("random_strength", *rs),
            "border_count": trial.suggest_int("border_count", int(bc[0]), int(bc[1])),
        }
    
    def to_ray_tune_space(self, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            from ray import tune
        except ImportError:
            raise ImportError("ray[tune] is required. Install with: pip install 'ray[tune]'")
        
        space = shrunk_space or {}
        
        lr = space.get("learning_rate", self.learning_rate)
        dp = space.get("depth", self.depth)
        l2 = space.get("l2_leaf_reg", self.l2_leaf_reg)
        bt = space.get("bagging_temperature", self.bagging_temperature)
        rs = space.get("random_strength", self.random_strength)
        bc = space.get("border_count", self.border_count)
        
        return {
            "learning_rate": tune.loguniform(*lr),
            "depth": tune.randint(int(dp[0]), int(dp[1]) + 1),
            "l2_leaf_reg": tune.uniform(*l2),
            "bagging_temperature": tune.uniform(*bt),
            "random_strength": tune.uniform(*rs),
            "border_count": tune.randint(int(bc[0]), int(bc[1]) + 1),
        }
