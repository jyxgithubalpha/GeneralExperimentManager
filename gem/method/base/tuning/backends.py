"""
TunerBackend - Pluggable tuning backends.

Supports:
- OptunaBackend: Serial/parallel Optuna search
- RayTuneBackend: Distributed Ray Tune search
- NNIBackend: NNI neural architecture search (for PyTorch)
"""


from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .search_space import BaseSearchSpace
from ...method_dataclasses import TuneResult


class TunerBackend(ABC):
    """
    Abstract base class for tuning backends.
    
    Subclasses implement optimize() using different search libraries.
    """
    
    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: "BaseSearchSpace",
        n_trials: int,
        direction: str,
        seed: int,
        warm_start_params: Optional[Dict[str, Any]] = None,
        shrunk_space: Optional[Dict[str, Any]] = None,
    ) -> "TuneResult":
        """
        Run optimization.
        
        Args:
            objective_fn: Function that takes params dict and returns metric value
            search_space: Search space definition
            n_trials: Number of trials
            direction: "maximize" or "minimize"
            seed: Random seed
            warm_start_params: Optional params to start with
            shrunk_space: Optional shrunk search space
            
        Returns:
            TuneResult with best parameters
        """
        pass


class OptunaBackend(TunerBackend):
    """
    Optuna-based tuning backend.
    
    Supports TPE sampler with warm start and parallel trials.
    """
    
    def __init__(
        self,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        show_progress_bar: bool = False,
    ):
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.show_progress_bar = show_progress_bar
    
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: "BaseSearchSpace",
        n_trials: int,
        direction: str,
        seed: int,
        warm_start_params: Optional[Dict[str, Any]] = None,
        shrunk_space: Optional[Dict[str, Any]] = None,
    ) -> "TuneResult":
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("optuna is required for OptunaBackend") from exc
        
        from ...method_dataclasses import TuneResult
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        
        warm_start_used = False
        if warm_start_params is not None:
            valid_params = {
                k: v for k, v in warm_start_params.items()
                if k in search_space.get_param_names()
            }
            if valid_params:
                try:
                    study.enqueue_trial(valid_params)
                    warm_start_used = True
                except Exception:
                    pass
        
        def trial_objective(trial) -> float:
            params = search_space.sample_optuna(trial, shrunk_space)
            return objective_fn(params)
        
        study.optimize(
            trial_objective,
            n_trials=n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=self.show_progress_bar,
        )
        
        all_trials = [
            {"params": t.params, "value": t.value, "state": str(t.state)}
            for t in study.trials
        ]
        
        return TuneResult(
            best_params=dict(study.best_trial.params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            all_trials=all_trials,
            warm_start_used=warm_start_used,
            shrunk_space_used=shrunk_space is not None,
        )


class RayTuneBackend(TunerBackend):
    """
    Ray Tune-based tuning backend.
    
    Supports distributed parallel search with Optuna integration.
    """
    
    def __init__(
        self,
        num_workers: int = 1,
        resources_per_trial: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ):
        self.num_workers = num_workers
        self.resources_per_trial = resources_per_trial or {"cpu": 1, "gpu": 0}
        self.verbose = verbose
    
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: "BaseSearchSpace",
        n_trials: int,
        direction: str,
        seed: int,
        warm_start_params: Optional[Dict[str, Any]] = None,
        shrunk_space: Optional[Dict[str, Any]] = None,
    ) -> "TuneResult":
        try:
            from ray import tune
            from ray.tune.search.optuna import OptunaSearch
        except ImportError as exc:
            raise ImportError(
                "ray[tune] is required. Install with: pip install 'ray[tune]'"
            ) from exc
        
        from ...method_dataclasses import TuneResult
        
        ray_space = search_space.to_ray_tune_space(shrunk_space)
        
        warm_start_used = False
        points_to_evaluate = None
        if warm_start_params is not None:
            valid_params = {
                k: v for k, v in warm_start_params.items()
                if k in search_space.get_param_names()
            }
            if valid_params:
                points_to_evaluate = [valid_params]
                warm_start_used = True
        
        mode = "max" if direction == "maximize" else "min"
        metric_name = "objective"
        
        optuna_search = OptunaSearch(
            metric=metric_name,
            mode=mode,
            seed=seed,
            points_to_evaluate=points_to_evaluate,
        )
        
        def trainable(config):
            score = objective_fn(config)
            return {metric_name: score}
        
        analysis = tune.run(
            trainable,
            config=ray_space,
            num_samples=n_trials,
            search_alg=optuna_search,
            resources_per_trial=self.resources_per_trial,
            verbose=self.verbose,
        )
        
        best_trial = analysis.get_best_trial(metric_name, mode)
        best_params = dict(best_trial.config)
        best_value = float(best_trial.last_result[metric_name])
        
        all_trials = [
            {
                "params": trial.config,
                "value": trial.last_result.get(metric_name),
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
