"""
GBDT Trainers - Trainers for LightGBM, XGBoost, CatBoost.
"""


from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...data.data_dataclasses import ProcessedViews
from ..base import BaseTrainer
from ..method_dataclasses import FitResult, RayDataViews, TrainConfig
from .adapters import LightGBMAdapter, XGBoostAdapter, CatBoostAdapter


class LightGBMTrainer(BaseTrainer):
    """LightGBM trainer."""
    
    def __init__(
        self,
        adapter: Optional[LightGBMAdapter] = None,
        use_ray_trainer: bool = False,
        ray_trainer_config: Optional[Dict[str, Any]] = None,
    ):
        self.adapter = adapter or LightGBMAdapter()
        self.use_ray_trainer = use_ray_trainer
        self.ray_trainer_config = ray_trainer_config or {}

    def fit(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        if config.use_ray_trainer or self.use_ray_trainer:
            raise NotImplementedError(
                "Ray trainer path is not enabled in this runtime. "
                "Use ExperimentConfig(use_ray=True) for split-level parallelism."
            )
        return self._fit_local(views, config, mode, sample_weights)

    @staticmethod
    def _import_lightgbm():
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm is required for LightGBMTrainer") from exc
        return lgb

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

    def _build_datasets(
        self,
        views: "ProcessedViews",
        sample_weights: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        weights = sample_weights or {}
        dtrain = self.adapter.to_dataset(views.train, weight=weights.get("train"))
        dval = self.adapter.to_dataset(views.val, reference=dtrain, weight=weights.get("val"))
        return dtrain, dval, {"train": dtrain, "val": dval}

    def _build_callbacks(self, lgb, config: TrainConfig, mode: str, evals_result: Dict[str, Dict[str, list]]):
        verbose = mode == "full"
        callbacks = [
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, first_metric_only=True, verbose=verbose),
            lgb.log_evaluation(period=config.verbose_eval if verbose else 0),
            lgb.record_evaluation(evals_result),
        ]
        return callbacks

    def _train_with_lgb(
        self, lgb, params: Dict[str, Any], dtrain: Any, dval: Any,
        config: TrainConfig, mode: str, feval_list=None,
    ) -> Tuple[Any, Dict[str, Dict[str, list]], int]:
        evals_result: Dict[str, Dict[str, list]] = {}
        callbacks = self._build_callbacks(lgb, config, mode, evals_result)

        model = lgb.train(
            params, dtrain,
            num_boost_round=config.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            feval=feval_list if feval_list else None,
            callbacks=callbacks,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.num_boost_round)
        return model, evals_result, best_iteration

    def _fit_local(
        self, views: "ProcessedViews", config: TrainConfig,
        mode: str = "full", sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        lgb = self._import_lightgbm()
        from ...utils.feval import FevalAdapterFactory
        from ...utils.objectives import ObjectiveFactory

        dtrain, dval, datasets = self._build_datasets(views, sample_weights)

        params = dict(config.params)
        params["seed"] = config.seed

        objective = ObjectiveFactory.get(config.objective_name, views=views, datasets=datasets)
        params["objective"] = objective

        split_views = {"train": views.train, "val": views.val, "test": views.test}
        feval_list = FevalAdapterFactory.create(config.feval_names, split_views, datasets)

        model, evals_result, best_iteration = self._train_with_lgb(
            lgb=lgb, params=params, dtrain=dtrain, dval=dval,
            config=config, mode=mode, feval_list=feval_list,
        )

        return FitResult(
            model=model, evals_result=evals_result, best_iteration=best_iteration,
            params=params, seed=config.seed,
        )

    def fit_from_ray_views(self, ray_views: RayDataViews, config: TrainConfig, mode: str = "full") -> FitResult:
        lgb = self._import_lightgbm()
        dtrain = self.adapter.from_ray_bundle(ray_views.train)
        dval = self.adapter.from_ray_bundle(ray_views.val, reference=dtrain)

        params = dict(config.params)
        params["seed"] = config.seed

        model, evals_result, best_iteration = self._train_with_lgb(
            lgb=lgb, params=params, dtrain=dtrain, dval=dval,
            config=config, mode=mode, feval_list=None,
        )

        return FitResult(
            model=model, evals_result=evals_result, best_iteration=best_iteration,
            params=params, seed=config.seed,
        )


class XGBoostTrainer(BaseTrainer):
    """XGBoost trainer."""
    
    def __init__(self, adapter: Optional[XGBoostAdapter] = None, use_gpu: Optional[bool] = None):
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
        self, views: "ProcessedViews", config: TrainConfig,
        mode: str = "full", sample_weights: Optional[Dict[str, Any]] = None,
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

    def _build_datasets(self, views: "ProcessedViews", sample_weights: Optional[Dict[str, Any]]) -> Tuple[Any, Any]:
        weights = sample_weights or {}
        dtrain = self.adapter.to_dataset(views.train, weight=weights.get("train"))
        dval = self.adapter.to_dataset(views.val, weight=weights.get("val"))
        return dtrain, dval

    @staticmethod
    def _resolve_best_iteration(model: Any, evals_result: Dict[str, Dict[str, list]], fallback: int) -> int:
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            best_iteration = int(best_iteration)
            if best_iteration >= 0:
                return min(best_iteration + 1, max(1, fallback))

        for split_metrics in evals_result.values():
            for series in split_metrics.values():
                if isinstance(series, list) and series:
                    return min(len(series), max(1, fallback))
        return max(1, fallback)

    def _fit_local(
        self, views: "ProcessedViews", config: TrainConfig,
        mode: str, sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        xgb = self._import_xgboost()

        dtrain, dval = self._build_datasets(views, sample_weights)
        params = dict(config.params)
        params["seed"] = config.seed
        if "eval_metric" not in params and config.feval_names:
            params["eval_metric"] = config.feval_names[0]
        params = self._apply_gpu_params(params)

        evals_result: Dict[str, Dict[str, list]] = {}
        model = xgb.train(
            params=params, dtrain=dtrain,
            num_boost_round=config.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=config.verbose_eval if mode == "full" else False,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.num_boost_round)
        return FitResult(
            model=model, evals_result=evals_result, best_iteration=best_iteration,
            params=params, seed=config.seed,
        )

    def fit_from_ray_views(self, ray_views: RayDataViews, config: TrainConfig, mode: str = "full") -> FitResult:
        xgb = self._import_xgboost()
        dtrain = self.adapter.from_ray_bundle(ray_views.train)
        dval = self.adapter.from_ray_bundle(ray_views.val)

        params = dict(config.params)
        params["seed"] = config.seed
        if "eval_metric" not in params and config.feval_names:
            params["eval_metric"] = config.feval_names[0]
        params = self._apply_gpu_params(params)

        evals_result: Dict[str, Dict[str, list]] = {}
        model = xgb.train(
            params=params, dtrain=dtrain,
            num_boost_round=config.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=config.verbose_eval if mode == "full" else False,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.num_boost_round)
        return FitResult(
            model=model, evals_result=evals_result, best_iteration=best_iteration,
            params=params, seed=config.seed,
        )


class CatBoostTrainer(BaseTrainer):
    """CatBoost trainer."""
    
    def __init__(self, adapter: Optional[CatBoostAdapter] = None, use_gpu: Optional[bool] = None):
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
        self, views: "ProcessedViews", config: TrainConfig,
        mode: str = "full", sample_weights: Optional[Dict[str, Any]] = None,
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

    def _build_datasets(self, views: "ProcessedViews", sample_weights: Optional[Dict[str, Any]]) -> Tuple[Any, Any]:
        weights = sample_weights or {}
        dtrain = self.adapter.to_dataset(views.train, weight=weights.get("train"))
        dval = self.adapter.to_dataset(views.val, weight=weights.get("val"))
        return dtrain, dval

    def _fit_local(
        self, views: "ProcessedViews", config: TrainConfig,
        mode: str, sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        CatBoostRegressor = self._import_catboost()

        dtrain, dval = self._build_datasets(views, sample_weights)

        params = dict(config.params)
        params["random_seed"] = config.seed
        params = self._apply_gpu_params(params)

        model = CatBoostRegressor(iterations=config.num_boost_round, **params)
        model.fit(
            dtrain, eval_set=dval,
            verbose=config.verbose_eval if mode == "full" else False,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        evals_result = self._normalize_evals_result(model.get_evals_result() or {})
        best_iteration = int(getattr(model, "get_best_iteration", lambda: 0)() or 0)
        if best_iteration <= 0:
            best_iteration = int(getattr(model, "tree_count_", 1) or 1)

        return FitResult(
            model=model, evals_result=evals_result, best_iteration=best_iteration,
            params=params, seed=config.seed,
        )

    def fit_from_ray_views(self, ray_views: RayDataViews, config: TrainConfig, mode: str = "full") -> FitResult:
        CatBoostRegressor = self._import_catboost()

        dtrain = self.adapter.from_ray_bundle(ray_views.train)
        dval = self.adapter.from_ray_bundle(ray_views.val)

        params = dict(config.params)
        params["random_seed"] = config.seed
        params = self._apply_gpu_params(params)

        model = CatBoostRegressor(iterations=config.num_boost_round, **params)
        model.fit(
            dtrain, eval_set=dval,
            verbose=config.verbose_eval if mode == "full" else False,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        evals_result = self._normalize_evals_result(model.get_evals_result() or {})
        best_iteration = int(getattr(model, "get_best_iteration", lambda: 0)() or 0)
        if best_iteration <= 0:
            best_iteration = int(getattr(model, "tree_count_", 1) or 1)

        return FitResult(
            model=model, evals_result=evals_result, best_iteration=best_iteration,
            params=params, seed=config.seed,
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
