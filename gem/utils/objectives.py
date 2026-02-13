"""
Objective factory for built-in and custom LightGBM objectives.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Union

import lightgbm as lgb
import numpy as np


class ObjectiveFactory:
    _registry: Dict[str, Callable] = {}
    _builtin: List[str] = [
        "regression",
        "regression_l2",
        "l2",
        "mse",
        "rmse",
        "regression_l1",
        "l1",
        "mae",
        "huber",
        "fair",
        "poisson",
        "binary",
        "multiclass",
        "cross_entropy",
        "lambdarank",
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
        print(f"Objective '{name}' not found, fallback to 'regression'.")
        return "regression"


@ObjectiveFactory.register("pearsonr_ic_loss")
def pearsonr_ic_loss_factory(**context) -> Callable:
    views = context["views"]
    datasets: Dict[str, lgb.Dataset] = context["datasets"]

    dataset_to_view = {
        id(datasets["train"]): views.train,
        id(datasets["val"]): views.val,
    }

    def objective(y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(dataset.label).ravel()

        view = dataset_to_view.get(id(dataset), views.train)
        if view.keys is None or "date" not in view.keys.columns:
            raise ValueError("pearsonr_ic_loss requires `date` column in view.keys.")

        dates = view.keys["date"].to_numpy()
        unique_dates = np.unique(dates)

        grad = np.zeros_like(y_pred, dtype=np.float64)
        hess = np.ones_like(y_pred, dtype=np.float64) * 1e-6

        n_days = max(1, len(unique_dates))

        for day in unique_dates:
            mask = dates == day
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue

            pred_day = y_pred[idx]
            true_day = y_true[idx]

            pred_mean = np.mean(pred_day)
            true_mean = np.mean(true_day)
            pred_centered = pred_day - pred_mean
            true_centered = true_day - true_mean

            pred_std = np.std(pred_day)
            true_std = np.std(true_day)

            if pred_std < 1e-8 or true_std < 1e-8:
                diff = pred_day - true_day
                grad[idx] = 2.0 * diff / n_days
                hess[idx] = 2.0
                continue

            cov_xy = np.mean(pred_centered * true_centered)
            grad_ic = (
                true_centered / (pred_std * true_std)
                - cov_xy * pred_centered / (pred_std ** 3 * true_std)
            ) / len(idx)

            grad[idx] = -grad_ic / n_days
            hess[idx] = np.abs(grad_ic) + 1e-6

        return grad, hess

    return objective
