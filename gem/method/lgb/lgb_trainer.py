"""
LightGBM trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews

from ..base import BaseTrainer
from ..method_dataclasses import FitResult, RayDataViews, TrainConfig
from .lgb_adapter import LightGBMAdapter


class LightGBMTrainer(BaseTrainer):
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
        dval = self.adapter.to_dataset(
            views.val,
            reference=dtrain,
            weight=weights.get("val"),
        )
        return dtrain, dval, {"train": dtrain, "val": dval}

    def _build_callbacks(self, lgb, config: TrainConfig, mode: str, evals_result: Dict[str, Dict[str, list]]):
        verbose = mode == "full"
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=config.early_stopping_rounds,
                first_metric_only=True,
                verbose=verbose,
            ),
            lgb.log_evaluation(period=config.verbose_eval if verbose else 0),
            lgb.record_evaluation(evals_result),
        ]
        return callbacks

    def _train_with_lgb(
        self,
        lgb,
        params: Dict[str, Any],
        dtrain: Any,
        dval: Any,
        config: TrainConfig,
        mode: str,
        feval_list=None,
    ) -> Tuple[Any, Dict[str, Dict[str, list]], int]:
        evals_result: Dict[str, Dict[str, list]] = {}
        callbacks = self._build_callbacks(lgb, config, mode, evals_result)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=config.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            feval=feval_list if feval_list else None,
            callbacks=callbacks,
        )

        best_iteration = self._resolve_best_iteration(model, evals_result, config.num_boost_round)
        return model, evals_result, best_iteration

    def _fit_local(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        lgb = self._import_lightgbm()

        from ...utils.feval import FevalAdapterFactory
        from ...utils.objectives import ObjectiveFactory

        dtrain, dval, datasets = self._build_datasets(views, sample_weights)

        params = dict(config.params)
        params["seed"] = config.seed

        objective = ObjectiveFactory.get(
            config.objective_name,
            views=views,
            datasets=datasets,
        )
        params["objective"] = objective

        split_views = {"train": views.train, "val": views.val, "test": views.test}
        feval_list = FevalAdapterFactory.create(config.feval_names, split_views, datasets)

        model, evals_result, best_iteration = self._train_with_lgb(
            lgb=lgb,
            params=params,
            dtrain=dtrain,
            dval=dval,
            config=config,
            mode=mode,
            feval_list=feval_list,
        )

        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_iteration,
            params=params,
            seed=config.seed,
        )

    def fit_from_ray_views(
        self,
        ray_views: RayDataViews,
        config: TrainConfig,
        mode: str = "full",
    ) -> FitResult:
        lgb = self._import_lightgbm()

        dtrain = self.adapter.from_ray_bundle(ray_views.train)
        dval = self.adapter.from_ray_bundle(ray_views.val, reference=dtrain)

        params = dict(config.params)
        params["seed"] = config.seed

        model, evals_result, best_iteration = self._train_with_lgb(
            lgb=lgb,
            params=params,
            dtrain=dtrain,
            dval=dval,
            config=config,
            mode=mode,
            feval_list=None,
        )

        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_iteration,
            params=params,
            seed=config.seed,
        )

    def _fit_with_ray(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        try:
            from ray.train import RunConfig, ScalingConfig
            from ray.train.lightgbm import LightGBMTrainer as RayLGBMTrainer
        except ImportError as exc:
            raise ImportError("ray[train] is required. Install with: pip install 'ray[train]'") from exc

        lgb = self._import_lightgbm()
        from ..base.base_adapter import RayDataAdapter

        weights = sample_weights or {}

        ray_views = RayDataAdapter.views_to_ray_views(views, weights)
        train_dataset = ray_views.train.to_ray_dataset()
        val_dataset = ray_views.val.to_ray_dataset()

        params = dict(config.params)
        params["seed"] = config.seed
        params["num_boost_round"] = config.num_boost_round

        scaling_config = ScalingConfig(
            num_workers=self.ray_trainer_config.get("num_workers", 1),
            use_gpu=self.ray_trainer_config.get("use_gpu", False),
        )
        run_config = RunConfig(name="lgb_train", verbose=0 if mode == "tune" else 1)

        ray_trainer = RayLGBMTrainer(
            params=params,
            label_column="y",
            datasets={"train": train_dataset, "valid": val_dataset},
            scaling_config=scaling_config,
            run_config=run_config,
        )

        result = ray_trainer.fit()

        model = None
        checkpoint = result.checkpoint
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                model_path = Path(checkpoint_dir) / "model.txt"
                if model_path.exists():
                    model = lgb.Booster(model_file=str(model_path))

        evals_result = {"train": {}, "val": {}}
        if result.metrics:
            for key, value in result.metrics.items():
                key_lower = key.lower()
                if "train" in key_lower:
                    metric_name = key.replace("train_", "")
                    evals_result["train"].setdefault(metric_name, []).append(value)
                elif "valid" in key_lower or "val" in key_lower:
                    metric_name = key.replace("valid_", "").replace("val_", "")
                    evals_result["val"].setdefault(metric_name, []).append(value)

        best_iteration = int(result.metrics.get("training_iteration", 1)) if result.metrics else 1

        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=max(1, best_iteration),
            params=params,
            seed=config.seed,
            checkpoint_path=checkpoint.path if checkpoint else None,
        )
