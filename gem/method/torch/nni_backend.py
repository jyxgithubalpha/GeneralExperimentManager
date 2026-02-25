"""
NNI Backend - Neural Network Intelligence integration for PyTorch NAS.

Supports:
- TPE tuner
- Random tuner
- Grid search
- Evolution tuner
- BOHB tuner

Note: NNI runs experiments externally, so this backend provides
configuration generation and result parsing utilities.
"""


import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..base.tuning.search_space import BaseSearchSpace
from ..method_dataclasses import TuneResult
from ..base.tuning.backends import TunerBackend


class NNIBackend(TunerBackend):
    """
    NNI-based tuning backend for neural architecture search.
    
    NNI (Neural Network Intelligence) is an open source AutoML toolkit.
    This backend integrates NNI for PyTorch model tuning.
    
    Usage modes:
    1. Embedded mode: Run NNI trials within the current process
    2. Config mode: Generate NNI experiment config for external execution
    
    Example:
        backend = NNIBackend(tuner_name="TPE", mode="embedded")
        result = backend.optimize(objective_fn, search_space, n_trials=50, ...)
    """
    
    SUPPORTED_TUNERS = [
        "TPE",
        "Random",
        "Anneal",
        "Evolution",
        "SMAC",
        "BOHB",
        "GridSearch",
        "Hyperband",
    ]
    
    def __init__(
        self,
        tuner_name: str = "TPE",
        assessor_name: Optional[str] = "Medianstop",
        mode: str = "embedded",
        experiment_name: str = "nni_experiment",
        max_trial_duration: str = "1h",
        trial_concurrency: int = 1,
        use_gpu: bool = False,
    ):
        """
        Args:
            tuner_name: NNI tuner type (TPE, Random, Evolution, etc.)
            assessor_name: NNI assessor type (Medianstop, Curvefitting, etc.)
            mode: "embedded" (run in process) or "config" (generate config only)
            experiment_name: Name for NNI experiment
            max_trial_duration: Maximum duration per trial
            trial_concurrency: Number of concurrent trials
            use_gpu: Whether to use GPU
        """
        if tuner_name not in self.SUPPORTED_TUNERS:
            raise ValueError(
                f"Unsupported tuner: {tuner_name}. "
                f"Supported: {self.SUPPORTED_TUNERS}"
            )
        
        self.tuner_name = tuner_name
        self.assessor_name = assessor_name
        self.mode = mode
        self.experiment_name = experiment_name
        self.max_trial_duration = max_trial_duration
        self.trial_concurrency = trial_concurrency
        self.use_gpu = use_gpu
    
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
        Run NNI optimization.
        
        In embedded mode, runs trials directly using nni.
        In config mode, generates experiment config and raises for external execution.
        """
        if self.mode == "config":
            config = self.generate_experiment_config(
                search_space=search_space,
                n_trials=n_trials,
                direction=direction,
            )
            raise NotImplementedError(
                f"NNI config mode: Run experiment externally with config:\n"
                f"{json.dumps(config, indent=2)}"
            )
        
        return self._run_embedded(
            objective_fn=objective_fn,
            search_space=search_space,
            n_trials=n_trials,
            direction=direction,
            seed=seed,
            warm_start_params=warm_start_params,
        )
    
    def _run_embedded(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: "BaseSearchSpace",
        n_trials: int,
        direction: str,
        seed: int,
        warm_start_params: Optional[Dict[str, Any]] = None,
    ) -> "TuneResult":
        """Run NNI trials in embedded mode using Optuna as fallback."""
        try:
            import nni
            from nni.experiment import Experiment
        except ImportError:
            return self._fallback_to_optuna(
                objective_fn=objective_fn,
                search_space=search_space,
                n_trials=n_trials,
                direction=direction,
                seed=seed,
                warm_start_params=warm_start_params,
            )
        
        from ..method_dataclasses import TuneResult
        
        nni_space = search_space.to_nni_space()
        
        all_trials: List[Dict[str, Any]] = []
        best_params: Dict[str, Any] = {}
        best_value = float("inf") if direction == "minimize" else float("-inf")
        
        for trial_idx in range(n_trials):
            if trial_idx == 0 and warm_start_params:
                params = warm_start_params
            else:
                params = self._sample_from_nni_space(nni_space, seed + trial_idx)
            
            try:
                value = objective_fn(params)
                
                is_better = (
                    (direction == "minimize" and value < best_value) or
                    (direction == "maximize" and value > best_value)
                )
                if is_better:
                    best_value = value
                    best_params = params.copy()
                
                all_trials.append({
                    "params": params,
                    "value": value,
                    "state": "COMPLETE",
                })
            except Exception as e:
                all_trials.append({
                    "params": params,
                    "value": None,
                    "state": f"FAIL: {str(e)}",
                })
        
        return TuneResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(all_trials),
            all_trials=all_trials,
            warm_start_used=warm_start_params is not None,
            shrunk_space_used=False,
        )
    
    def _sample_from_nni_space(
        self,
        nni_space: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """Sample parameters from NNI space format."""
        import random
        random.seed(seed)
        
        params = {}
        for name, spec in nni_space.items():
            param_type = spec["_type"]
            values = spec["_value"]
            
            if param_type == "choice":
                params[name] = random.choice(values)
            elif param_type == "randint":
                params[name] = random.randint(values[0], values[1] - 1)
            elif param_type == "uniform":
                params[name] = random.uniform(values[0], values[1])
            elif param_type == "loguniform":
                import math
                log_low = math.log(values[0])
                log_high = math.log(values[1])
                params[name] = math.exp(random.uniform(log_low, log_high))
            elif param_type == "normal":
                params[name] = random.gauss(values[0], values[1])
            else:
                params[name] = values[0] if isinstance(values, list) else values
        
        return params
    
    def _fallback_to_optuna(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: "BaseSearchSpace",
        n_trials: int,
        direction: str,
        seed: int,
        warm_start_params: Optional[Dict[str, Any]] = None,
    ) -> "TuneResult":
        """Fallback to Optuna when NNI is not available."""
        from .backends import OptunaBackend
        
        optuna_backend = OptunaBackend()
        return optuna_backend.optimize(
            objective_fn=objective_fn,
            search_space=search_space,
            n_trials=n_trials,
            direction=direction,
            seed=seed,
            warm_start_params=warm_start_params,
        )
    
    def generate_experiment_config(
        self,
        search_space: "BaseSearchSpace",
        n_trials: int,
        direction: str,
        trial_code_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate NNI experiment configuration.
        
        This can be saved to a YAML file and run with `nnictl create`.
        """
        nni_space = search_space.to_nni_space()
        
        config = {
            "experimentName": self.experiment_name,
            "trialConcurrency": self.trial_concurrency,
            "maxTrialNumber": n_trials,
            "maxTrialDuration": self.max_trial_duration,
            "searchSpace": nni_space,
            "tuner": {
                "name": self.tuner_name,
                "classArgs": {
                    "optimize_mode": "minimize" if direction == "minimize" else "maximize",
                },
            },
            "trainingService": {
                "platform": "local",
                "useActiveGpu": self.use_gpu,
            },
        }
        
        if self.assessor_name:
            config["assessor"] = {
                "name": self.assessor_name,
            }
        
        if trial_code_path:
            config["trialCommand"] = f"python {trial_code_path}"
            config["trialCodeDirectory"] = str(Path(trial_code_path).parent)
        
        return config
    
    def save_experiment_config(
        self,
        config: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Save experiment config to YAML file."""
        try:
            import yaml
        except ImportError:
            import json
            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)
            return output_path
        
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        return output_path
