"""
CatBoost trainer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews

from ..base import BaseTrainer
from ..method_dataclasses import FitResult, TrainConfig, RayDataViews
from .cat_adapter import CatBoostAdapter


class CatBoostTrainer(BaseTrainer):
    def __init__(
        self,
        adapter: Optional[CatBoostAdapter] = None,
        use_gpu: Optional[bool] = None,
    ):
        self.adapter = adapter or CatBoostAdapter()
        self.use_gpu = use_gpu

    @staticmethod
    def _import_catboost():
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("catboost is required for CatBoostTrainer") from exc
        return CatBoostRegressor

    def fit(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        return self._fit_local(views, config, mode, sample_weights)

    def _apply_gpu_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = bool(params.get("use_gpu", False))
        params.pop("use_gpu", None)
        if use_gpu:
            params.setdefault("task_type", "GPU")
        return params

    def _build_datasets(
        self,
        views: "ProcessedViews",
        sample_weights: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any]:
        weights = sample_weights or {}
        dtrain = self.adapter.to_dataset(views.train, weight=weights.get("train"))
        dval = self.adapter.to_dataset(views.val, weight=weights.get("val"))
        return dtrain, dval

    def _fit_local(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str,
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        CatBoostRegressor = self._import_catboost()
        start_time = time.time()

        dtrain, dval = self._build_datasets(views, sample_weights)

        params = dict(config.params)
        params["random_seed"] = config.seed
        params = self._apply_gpu_params(params)

        model = CatBoostRegressor(
            iterations=config.num_boost_round,
            **params,
        )
        model.fit(
            dtrain,
            eval_set=dval,
            verbose=config.verbose_eval if mode == "full" else False,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        evals_result = self._normalize_evals_result(model.get_evals_result() or {})
        best_iteration = int(getattr(model, "get_best_iteration", lambda: 0)() or 0)
        if best_iteration <= 0:
            best_iteration = int(getattr(model, "tree_count_", 1) or 1)

        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_iteration,
            params=params,
            seed=config.seed,
            train_time=time.time() - start_time,
        )

    def fit_from_ray_views(
        self,
        ray_views: RayDataViews,
        config: TrainConfig,
        mode: str = "full",
    ) -> FitResult:
        CatBoostRegressor = self._import_catboost()
        start_time = time.time()

        dtrain = self.adapter.from_ray_bundle(ray_views.train)
        dval = self.adapter.from_ray_bundle(ray_views.val)

        params = dict(config.params)
        params["random_seed"] = config.seed
        params = self._apply_gpu_params(params)

        model = CatBoostRegressor(
            iterations=config.num_boost_round,
            **params,
        )
        model.fit(
            dtrain,
            eval_set=dval,
            verbose=config.verbose_eval if mode == "full" else False,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        evals_result = self._normalize_evals_result(model.get_evals_result() or {})
        best_iteration = int(getattr(model, "get_best_iteration", lambda: 0)() or 0)
        if best_iteration <= 0:
            best_iteration = int(getattr(model, "tree_count_", 1) or 1)

        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_iteration,
            params=params,
            seed=config.seed,
            train_time=time.time() - start_time,
        )

    @staticmethod
    def _normalize_evals_result(evals_result: Dict[str, Dict[str, list]]) -> Dict[str, Dict[str, list]]:
        if not evals_result:
            return {}

        normalized: Dict[str, Dict[str, list]] = {}
        if "learn" in evals_result:
            normalized["train"] = evals_result["learn"]
        if "validation" in evals_result:
            normalized["val"] = evals_result["validation"]
        if "validation_0" in evals_result and "val" not in normalized:
            normalized["val"] = evals_result["validation_0"]
        return normalized or evals_result
