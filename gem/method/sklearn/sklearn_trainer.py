"""
Generic sklearn trainer for tabular regression models.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Mapping

from hydra.utils import instantiate

from ...data.data_dataclasses import ProcessedViews
from ..base import BaseTrainer
from ..method_dataclasses import FitResult, TrainConfig


class SklearnTrainer(BaseTrainer):
    def __init__(self, model_config: Optional[Any] = None):
        if model_config is None:
            raise ValueError("model_config is required for SklearnTrainer.")
        self.model_config = model_config

    def fit(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        model = self._build_model(config)

        X_train = views.train.X
        y_train = views.train.y.ravel() if views.train.y.ndim > 1 else views.train.y
        weights = (sample_weights or {}).get("train")

        fit_kwargs: Dict[str, Any] = {}
        if weights is not None and self._supports_arg(model.fit, "sample_weight"):
            fit_kwargs["sample_weight"] = weights

        model.fit(X_train, y_train, **fit_kwargs)

        return FitResult(
            model=model,
            evals_result={"train": {}, "val": {}},
            best_iteration=1,
            params=dict(config.params),
            seed=config.seed,
        )

    def _build_model(self, config: TrainConfig):
        params = dict(config.params or {})
        if self._config_supports_random_state() and "random_state" not in params:
            params["random_state"] = config.seed
        return instantiate(self.model_config, **params)

    def _config_supports_random_state(self) -> bool:
        try:
            if isinstance(self.model_config, Mapping) and "_target_" in self.model_config:
                target = self.model_config["_target_"]
                cls = self._resolve_class(target)
                return self._supports_arg(cls.__init__, "random_state")
        except Exception:
            return False
        return False

    @staticmethod
    def _resolve_class(target: str):
        module_path, _, class_name = target.rpartition(".")
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    @staticmethod
    def _supports_arg(func, arg_name: str) -> bool:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False
        return arg_name in signature.parameters
