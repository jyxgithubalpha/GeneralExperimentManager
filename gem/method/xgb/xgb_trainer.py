"""
XGBoost trainer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews

from ..base import BaseTrainer
from ..method_dataclasses import FitResult, TrainConfig, RayDataViews
from .xgb_adapter import XGBoostAdapter


class XGBoostTrainer(BaseTrainer):
    def __init__(
        self,
        adapter: Optional[XGBoostAdapter] = None,
        use_gpu: Optional[bool] = None,
    ):
        self.adapter = adapter or XGBoostAdapter()
        self.use_gpu = use_gpu

    @staticmethod
    def _import_xgboost():
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostTrainer") from exc
        return xgb

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
            params.setdefault("tree_method", "gpu_hist")
            params.setdefault("predictor", "gpu_predictor")
            params.setdefault("device", "cuda")
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

    @staticmethod
    def _resolve_best_iteration(model: Any, evals_result: Dict[str, Dict[str, list]], fallback: int) -> int:
        best_iteration = int(getattr(model, "best_iteration", 0) or 0)
        if best_iteration > 0:
            return min(best_iteration, max(1, fallback))

        for split_metrics in evals_result.values():
            for series in split_metrics.values():
                if isinstance(series, list) and series:
                    return min(len(series), max(1, fallback))
        return max(1, fallback)

    def _fit_local(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str,
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        xgb = self._import_xgboost()
        start_time = time.time()

        dtrain, dval = self._build_datasets(views, sample_weights)
        params = dict(config.params)
        params["seed"] = config.seed
        params = self._apply_gpu_params(params)

        evals_result: Dict[str, Dict[str, list]] = {}
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=config.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=config.verbose_eval if mode == "full" else False,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.num_boost_round)
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
        xgb = self._import_xgboost()
        start_time = time.time()

        dtrain = self.adapter.from_ray_bundle(ray_views.train)
        dval = self.adapter.from_ray_bundle(ray_views.val)

        params = dict(config.params)
        params["seed"] = config.seed
        params = self._apply_gpu_params(params)

        evals_result: Dict[str, Dict[str, list]] = {}
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=config.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=config.verbose_eval if mode == "full" else False,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.num_boost_round)
        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_iteration,
            params=params,
            seed=config.seed,
            train_time=time.time() - start_time,
        )
