"""
目标函数工厂 - 自定义损失函数
"""



from typing import Any, Callable, Dict, List, Tuple, Union

import lightgbm as lgb
import numpy as np

from ..data.data_dataclasses import SplitSpec
from ..data.data_dataclasses import SplitData


class ObjectiveFactory:
    """目标函数工厂"""
    _registry: Dict[str, Callable] = {}
    _builtin: List[str] = [
        "regression", "regression_l2", "l2", "mse", "rmse",
        "regression_l1", "l1", "mae", "huber", "fair", "poisson",
        "binary", "multiclass", "cross_entropy", "lambdarank"
    ]
    
    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str, **context) -> Union[str, Callable]:
        if name in cls._builtin:
            return name
        if name in cls._registry:
            return cls._registry[name](**context)
        print(f"Objective '{name}' not found, using 'regression'")
        return "regression"


@ObjectiveFactory.register("pearsonr_ic_loss")
def pearsonr_ic_loss_factory(**context) -> Callable:
    """Pearson IC损失函数"""
    split_data: SplitData = context["split_data"]
    datasets: Dict[str, lgb.Dataset] = context["datasets"]
    dataset_to_bundle = {id(ds): name for name, ds in datasets.items()}
    
    def objective(y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(dataset.label).ravel()
        
        bundle_name = dataset_to_bundle.get(id(dataset), "train")
        bundle = split_data.get(bundle_name)
        date = bundle.meta["date"].values
        
        n = len(y_pred)
        grad = np.zeros(n, dtype=np.float64)
        hess = np.ones(n, dtype=np.float64) * 1e-6
        
        for d in np.unique(date):
            mask = date == d
            idx = np.where(mask)[0]
            n_d = len(idx)
            if n_d < 2:
                continue
            
            pred_d, true_d = y_pred[idx], y_true[idx]
            pred_mean, true_mean = np.mean(pred_d), np.mean(true_d)
            pred_centered = pred_d - pred_mean
            true_centered = true_d - true_mean
            pred_std, true_std = np.std(pred_d), np.std(true_d)
            
            if pred_std < 1e-8 or true_std < 1e-8:
                diff = pred_d - true_d
                grad[idx] = 2 * diff / len(np.unique(date))
                hess[idx] = 2 * np.ones(n_d)
                continue
            
            cov_xy = np.mean(pred_centered * true_centered)
            grad_ic = (true_centered / (pred_std * true_std) - 
                      cov_xy * pred_centered / (pred_std**3 * true_std)) / n_d
            grad[idx] = -grad_ic / len(np.unique(date))
            hess[idx] = np.abs(grad_ic) + 1e-6
        
        return grad, hess
    
    return objective
