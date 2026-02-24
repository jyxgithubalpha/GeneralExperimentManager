"""
XGBoost hyper-parameter tuner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ...experiment.experiment_dataclasses import RollingState, TuningState

from ..base import BaseTrainer, BaseTuner
from ..method_dataclasses import TrainConfig, TuneResult
from .xgb_param_space import XGBoostParamSpace


class XGBoostTuner(BaseTuner):
    def __init__(
        self,
        param_space: Optional[XGBoostParamSpace] = None,
        base_params: Optional[Dict[str, Any]] = None,
        n_trials: int = 50,
        target_metric: str = "pearsonr_ic",
        seed: int = 42,
        direction: str = "maximize",
        use_shrinkage: bool = True,
        shrink_ratio: float = 0.5,
        use_ray_tune: bool = False,
        parallel_trials: int = 1,
        use_warm_start: bool = True,
    ):
        self.param_space = param_space or XGBoostParamSpace()
        self.base_params = base_params or {}
        self.n_trials = n_trials
        self.target_metric = target_metric
        self.seed = seed
        self.direction = direction
        self.use_shrinkage = use_shrinkage
        self.shrink_ratio = shrink_ratio
        self.use_ray_tune = use_ray_tune
        self.parallel_trials = parallel_trials
        self.use_warm_start = use_warm_start

        self._last_best_params: Optional[Dict[str, Any]] = None
        self._last_best_value: Optional[float] = None

    def tune(
        self,
        views: "ProcessedViews",
        trainer: BaseTrainer,
        config: TrainConfig,
        tuning_state: Optional["TuningState"] = None,
        rolling_state: Optional["RollingState"] = None,
    ) -> TuneResult:
        if tuning_state is None and rolling_state is not None:
            from ...experiment.experiment_dataclasses import TuningState as TS

            tuning_state = rolling_state.get_state(TS)

        if self.use_ray_tune and self.parallel_trials > 1:
            return self._tune_with_ray(views, trainer, config, tuning_state)
        return self._tune_with_optuna(views, trainer, config, tuning_state)

    @staticmethod
    def _extract_validation_metric(fit_result, metric_name: str) -> float:
        val_metrics = fit_result.evals_result.get("val", {})
        if metric_name not in val_metrics:
            raise ValueError(
                f"Metric '{metric_name}' not found in val evals_result. "
                f"Available: {list(val_metrics.keys())}"
            )

        scores = val_metrics[metric_name]
        if not scores:
            raise ValueError(f"Validation metric '{metric_name}' has empty score list.")

        best_iteration = max(1, int(fit_result.best_iteration))
        best_index = min(best_iteration, len(scores)) - 1
        return float(scores[best_index])

    def _build_shrunk_space(self, tuning_state: Optional["TuningState"]):
        if self.use_shrinkage and tuning_state is not None:
            return tuning_state.get_shrunk_space(self.param_space.to_dict())
        return None

    def _build_warm_start_points(self, tuning_state: Optional["TuningState"]):
        if not self.use_warm_start or tuning_state is None:
            return None
        if tuning_state.last_best_params is None:
            return None

        warm_params = {
            key: value
            for key, value in tuning_state.last_best_params.items()
            if key in self.param_space.get_param_names()
        }
        if not warm_params:
            return None
        return warm_params

    def _tune_with_optuna(
        self,
        views: "ProcessedViews",
        trainer: BaseTrainer,
        config: TrainConfig,
        tuning_state: Optional["TuningState"] = None,
    ) -> TuneResult:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("optuna is required for XGBoostTuner") from exc

        shrunk_space = self._build_shrunk_space(tuning_state)

        def objective(trial) -> float:
            sampled_params = self.param_space.sample(trial, shrunk_space)
            params = {**self.base_params, **sampled_params}
            tune_config = config.for_tuning(params, self.seed)
            fit_result = trainer.fit(views, tune_config, mode="tune")
            metric_name = tune_config.feval_names[0]
            return self._extract_validation_metric(fit_result, metric_name)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        warm_start_used = False
        warm_point = self._build_warm_start_points(tuning_state)
        if warm_point is not None:
            try:
                study.enqueue_trial(warm_point)
                warm_start_used = True
            except Exception:
                warm_start_used = False

        study.optimize(objective, n_trials=self.n_trials)

        best_params = {**self.base_params, **study.best_trial.params}
        best_value = float(study.best_value)

        self._last_best_params = best_params
        self._last_best_value = best_value

        all_trials = [
            {"params": t.params, "value": t.value, "state": str(t.state)}
            for t in study.trials
        ]

        return TuneResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(study.trials),
            all_trials=all_trials,
            warm_start_used=warm_start_used,
            shrunk_space_used=shrunk_space is not None,
        )

    def _tune_with_ray(
        self,
        views: "ProcessedViews",
        trainer: BaseTrainer,
        config: TrainConfig,
        tuning_state: Optional["TuningState"] = None,
    ) -> TuneResult:
        try:
            from ray import tune
            from ray.tune.search.optuna import OptunaSearch
        except ImportError as exc:
            raise ImportError("ray[tune] is required. Install with: pip install 'ray[tune]'") from exc

        try:
            import optuna  # noqa: F401
        except ImportError as exc:
            raise ImportError("optuna is required for XGBoostTuner") from exc

        shrunk_space = self._build_shrunk_space(tuning_state)
        ray_search_space = self.param_space.to_ray_tune_space(shrunk_space)

        warm_start_used = False
        points_to_evaluate = None
        warm_point = self._build_warm_start_points(tuning_state)
        if warm_point is not None:
            points_to_evaluate = [warm_point]
            warm_start_used = True

        optuna_search = OptunaSearch(
            metric=self.target_metric,
            mode="max" if self.direction == "maximize" else "min",
            seed=self.seed,
            points_to_evaluate=points_to_evaluate,
        )

        def trainable(ray_config):
            params = {**self.base_params, **ray_config}
            tune_config = config.for_tuning(params, self.seed)
            fit_result = trainer.fit(views, tune_config, mode="tune")
            metric_name = tune_config.feval_names[0]
            score = self._extract_validation_metric(fit_result, metric_name)
            return {self.target_metric: score}

        analysis = tune.run(
            trainable,
            config=ray_search_space,
            num_samples=self.n_trials,
            search_alg=optuna_search,
            resources_per_trial={"cpu": 1, "gpu": 0},
            verbose=0,
        )

        best_trial = analysis.get_best_trial(
            self.target_metric,
            "max" if self.direction == "maximize" else "min",
        )
        best_params = {**self.base_params, **best_trial.config}
        best_value = float(best_trial.last_result[self.target_metric])

        self._last_best_params = best_params
        self._last_best_value = best_value

        all_trials = [
            {
                "params": trial.config,
                "value": trial.last_result.get(self.target_metric),
                "state": str(trial.status),
            }
            for trial in analysis.trials
        ]

        return TuneResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(analysis.trials),
            all_trials=all_trials,
            warm_start_used=warm_start_used,
            shrunk_space_used=shrunk_space is not None,
        )

    @property
    def last_best_params(self) -> Optional[Dict[str, Any]]:
        return self._last_best_params

    @property
    def last_best_value(self) -> Optional[float]:
        return self._last_best_value
