"""
UnifiedTuner - Framework-agnostic hyperparameter/architecture tuner.

Combines SearchSpace + TunerBackend for flexible tuning.
"""


from typing import Any, Dict, Optional

from ....data.data_dataclasses import ProcessedViews
from ....experiment.states import RollingState, TuningState
from ...method_dataclasses import TrainConfig, TuneResult
from ..base_trainer import BaseTrainer
from .backends import OptunaBackend, TunerBackend
from .search_space import BaseSearchSpace


class UnifiedTuner:
    """
    Unified tuner supporting all frameworks and backends.
    
    Composes SearchSpace + TunerBackend for maximum flexibility.
    
    Example:
        # GBDT + Optuna
        tuner = UnifiedTuner(
            search_space=LightGBMSpace(),
            backend=OptunaBackend(),
            target_metric="pearsonr_ic",
            direction="maximize",
        )
        
        # PyTorch + NNI
        tuner = UnifiedTuner(
            search_space=MLPArchSpace(),
            backend=NNIBackend(),
            target_metric="val_mse",
            direction="minimize",
        )
    """
    
    def __init__(
        self,
        search_space: BaseSearchSpace,
        backend: Optional[TunerBackend] = None,
        base_params: Optional[Dict[str, Any]] = None,
        n_trials: int = 50,
        target_metric: str = "pearsonr_ic",
        direction: str = "maximize",
        seed: int = 42,
        use_warm_start: bool = True,
        use_shrinkage: bool = True,
        shrink_ratio: float = 0.5,
    ):
        """
        Args:
            search_space: Search space definition
            backend: Tuner backend (default: OptunaBackend)
            base_params: Base parameters to merge with sampled params
            n_trials: Number of trials
            target_metric: Metric name to optimize
            direction: "maximize" or "minimize"
            seed: Random seed
            use_warm_start: Use warm start from previous best params
            use_shrinkage: Shrink search space based on previous results
            shrink_ratio: Ratio to shrink search space
        """
        self.search_space = search_space
        self.backend = backend or OptunaBackend()
        self.base_params = base_params or {}
        self.n_trials = n_trials
        self.target_metric = target_metric
        self.direction = direction
        self.seed = seed
        self.use_warm_start = use_warm_start
        self.use_shrinkage = use_shrinkage
        self.shrink_ratio = shrink_ratio
        
        self._last_best_params: Optional[Dict[str, Any]] = None
        self._last_best_value: Optional[float] = None
    
    def tune(
        self,
        views: "ProcessedViews",
        trainer: "BaseTrainer",
        config: "TrainConfig",
        tuning_state: Optional["TuningState"] = None,
        rolling_state: Optional["RollingState"] = None,
    ) -> "TuneResult":
        """
        Run hyperparameter/architecture search.
        
        Args:
            views: Processed data views
            trainer: Trainer instance
            config: Training configuration
            tuning_state: Optional tuning state for warm start
            rolling_state: Optional rolling state
            
        Returns:
            TuneResult with best parameters
        """
        if tuning_state is None and rolling_state is not None:
            from ....experiment.states import TuningState as TS
            tuning_state = rolling_state.get_state(TS)
        
        warm_params = self._get_warm_start_params(tuning_state)
        shrunk_space = self._get_shrunk_space(tuning_state)
        
        def objective(params: Dict[str, Any]) -> float:
            merged_params = {**self.base_params, **params}
            tune_config = config.for_tuning(merged_params, self.seed)
            fit_result = trainer.fit(views, tune_config, mode="tune")
            return self._extract_metric(fit_result, tune_config)
        
        result = self.backend.optimize(
            objective_fn=objective,
            search_space=self.search_space,
            n_trials=self.n_trials,
            direction=self.direction,
            seed=self.seed,
            warm_start_params=warm_params,
            shrunk_space=shrunk_space,
        )
        
        result.best_params = {**self.base_params, **result.best_params}
        
        self._last_best_params = result.best_params
        self._last_best_value = result.best_value
        
        return result
    
    def _get_warm_start_params(
        self,
        tuning_state: Optional["TuningState"],
    ) -> Optional[Dict[str, Any]]:
        """Get warm start parameters from tuning state."""
        if not self.use_warm_start or tuning_state is None:
            return None
        if tuning_state.last_best_params is None:
            return None
        
        param_names = set(self.search_space.get_param_names())
        warm_params = {
            k: v for k, v in tuning_state.last_best_params.items()
            if k in param_names
        }
        return warm_params if warm_params else None
    
    def _get_shrunk_space(
        self,
        tuning_state: Optional["TuningState"],
    ) -> Optional[Dict[str, Any]]:
        """Get shrunk search space based on previous best params."""
        if not self.use_shrinkage or tuning_state is None:
            return None
        if tuning_state.last_best_params is None:
            return None
        
        return self.search_space.get_shrunk_space(
            tuning_state.last_best_params,
            self.shrink_ratio,
        )
    
    def _extract_metric(
        self,
        fit_result,
        config: "TrainConfig",
    ) -> float:
        """Extract target metric from fit result."""
        metric_name = config.feval_names[0] if config.feval_names else self.target_metric
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
    
    @property
    def last_best_params(self) -> Optional[Dict[str, Any]]:
        return self._last_best_params
    
    @property
    def last_best_value(self) -> Optional[float]:
        return self._last_best_value
