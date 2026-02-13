"""
Method related data classes

Contains:
- TransformStats: Transform statistics (thresholds, mean, std, etc.)
- TransformState: Transform state
- RayDataBundle: Ray Data bundle
- RayDataViews: Ray Data view collection
- TuneConfig: Search configuration
- TrainConfig: Training configuration
- TuneResult: Search result
- FitResult: Training result
- EvalResult: Evaluation result
- MethodOutput: Method complete output
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


# =============================================================================
# Transform related
# =============================================================================

@dataclass
class TransformStats:
    """
    Transform statistics - computed from train/val
    
    Contains X and y:
    - Thresholds (quantiles)
    - Mean
    - Standard deviation (std)
    - Other custom statistics
    """
    # X statistics
    X_mean: Optional[np.ndarray] = None
    X_std: Optional[np.ndarray] = None
    X_lower_quantile: Optional[np.ndarray] = None
    X_upper_quantile: Optional[np.ndarray] = None
    X_median: Optional[np.ndarray] = None
    
    # y statistics
    y_mean: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None
    y_lower_quantile: Optional[np.ndarray] = None
    y_upper_quantile: Optional[np.ndarray] = None
    y_median: Optional[np.ndarray] = None
    
    # Custom statistics
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "X_lower_quantile": self.X_lower_quantile,
            "X_upper_quantile": self.X_upper_quantile,
            "X_median": self.X_median,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "y_lower_quantile": self.y_lower_quantile,
            "y_upper_quantile": self.y_upper_quantile,
            "y_median": self.y_median,
            **self.custom,
        }


@dataclass
class TransformState:
    """
    Transform state - for storing statistics computed during fit
    
    Supports state storage for chained transforms
    """
    stats: Dict[str, Any] = field(default_factory=dict)
    transform_stats: Optional[TransformStats] = None

# =============================================================================
# Ray Data related
# =============================================================================

@dataclass
class RayDataBundle:
    """
    Ray Data bundle - for Ray distributed training
    
    Stores numpy arrays and ray.data.Dataset converted from pl.DataFrame
    """
    # Numpy data
    X: np.ndarray
    y: np.ndarray
    keys: Optional[np.ndarray] = None
    sample_weight: Optional[np.ndarray] = None
    
    # Meta information
    feature_names: Optional[List[str]] = None
    label_names: Optional[List[str]] = None
    n_samples: int = 0
    n_features: int = 0
    
    # Ray Data (lazy creation)
    _ray_dataset: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1] if self.X.ndim > 1 else 1
    
    def to_ray_dataset(self, include_weight: bool = True) -> Any:
        """
        Convert to ray.data.Dataset
        
        Returns:
            ray.data.Dataset
        """
        try:
            import ray.data
        except ImportError:
            raise ImportError("ray[data] is required. Install with: pip install 'ray[data]'")
        
        if self._ray_dataset is not None:
            return self._ray_dataset
        
        # Build data dictionary
        data_dict = {
            "X": self.X,
            "y": self.y,
        }
        if self.keys is not None:
            data_dict["keys"] = self.keys
        if include_weight and self.sample_weight is not None:
            data_dict["sample_weight"] = self.sample_weight
        
        self._ray_dataset = ray.data.from_numpy(data_dict)
        return self._ray_dataset


@dataclass
class RayDataViews:
    """
    Ray Data view collection - train/val/test
    """
    train: RayDataBundle
    val: RayDataBundle
    test: RayDataBundle
    transform_state: Optional[TransformState] = None
    transform_stats: Optional[TransformStats] = None


# =============================================================================
# Training configuration related
# =============================================================================

@dataclass
class TuneConfig:
    """
    Hyperparameter search configuration
    
    Attributes:
        n_trials: Number of search trials
        timeout: Timeout in seconds
        target_metric: Target metric
        direction: Optimization direction (maximize/minimize)
        use_ray_tune: Whether to use Ray Tune
        use_optuna: Whether to use Optuna
        parallel_trials: Number of parallel trials
        use_warm_start: Whether to use historical hyperparameters as starting point
        shrink_ratio: Search space shrink ratio
    """
    n_trials: int = 50
    timeout: Optional[int] = None
    target_metric: str = "pearsonr_ic"
    direction: str = "maximize"
    use_ray_tune: bool = False
    use_optuna: bool = True
    parallel_trials: int = 1
    use_warm_start: bool = True
    shrink_ratio: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_trials < 0:
            raise ValueError(f"n_trials must be >= 0, got {self.n_trials}")
        if self.parallel_trials <= 0:
            raise ValueError(
                f"parallel_trials must be > 0, got {self.parallel_trials}"
            )
        if self.shrink_ratio < 0:
            raise ValueError(f"shrink_ratio must be >= 0, got {self.shrink_ratio}")


@dataclass
class TrainConfig:
    """
    Training configuration
    
    Attributes:
        params: Model hyperparameters
        num_boost_round: Maximum iterations
        early_stopping_rounds: Early stopping rounds
        feval_names: Evaluation metric name list
        objective_name: Objective function name
        seed: Random seed
        verbose_eval: Log printing frequency
        use_ray_trainer: Whether to use Ray Trainer
    """
    params: Dict[str, Any]
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    feval_names: List[str] = field(default_factory=lambda: ["pearsonr_ic"])
    objective_name: str = "regression"
    seed: int = 42
    verbose_eval: int = 100
    use_ray_trainer: bool = False

    def __post_init__(self) -> None:
        if self.num_boost_round <= 0:
            raise ValueError(
                f"num_boost_round must be > 0, got {self.num_boost_round}"
            )
        if self.early_stopping_rounds <= 0:
            raise ValueError(
                "early_stopping_rounds must be > 0, "
                f"got {self.early_stopping_rounds}"
            )
        if not self.feval_names:
            raise ValueError("feval_names must not be empty.")
    
    def with_params(self, **kwargs) -> "TrainConfig":
        """Return new configuration with some parameters overridden"""
        new_params = {**self.params, **kwargs}
        return TrainConfig(
            params=new_params,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            feval_names=self.feval_names,
            objective_name=self.objective_name,
            seed=self.seed,
            verbose_eval=self.verbose_eval,
            use_ray_trainer=self.use_ray_trainer,
        )
    
    def for_tuning(self, params: Dict[str, Any], seed: int) -> "TrainConfig":
        """Create lightweight configuration for tuning"""
        return TrainConfig(
            params=params,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            feval_names=self.feval_names[:1],  # Use only first metric
            objective_name=self.objective_name,
            seed=seed,
            verbose_eval=0,
            use_ray_trainer=False,  # Don't use Ray Trainer during tuning
        )


# =============================================================================
# Training result related
# =============================================================================

@dataclass
class TuneResult:
    """
    Hyperparameter search result
    
    Attributes:
        best_params: Best hyperparameters
        best_value: Best objective value
        n_trials: Number of completed trials
        all_trials: All trial results
        search_time: Search time taken
        warm_start_used: Whether warm start was used
        shrunk_space_used: Whether shrunk space was used
    """
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    all_trials: Optional[List[Dict[str, Any]]] = None
    search_time: float = 0.0
    warm_start_used: bool = False
    shrunk_space_used: bool = False


@dataclass
class FitResult:
    """
    Training result
    
    Attributes:
        model: Trained model
        evals_result: Evaluation result history
        best_iteration: Best iteration number
        params: Used hyperparameters
        seed: Random seed
        train_time: Training time taken
        feature_importance: Feature importance DataFrame
        checkpoint_path: Ray checkpoint path (if using Ray Trainer)
    """
    model: Any  # lgb.Booster or other
    evals_result: Dict[str, Dict[str, List[float]]]
    best_iteration: int
    params: Dict[str, Any]
    seed: int
    train_time: float = 0.0
    feature_importance: Optional[pl.DataFrame] = None
    checkpoint_path: Optional[Path] = None


@dataclass 
class EvalResult:
    """
    Evaluation result
    
    Attributes:
        metrics: Metric dictionary (without mode prefix)
        series: Time series metrics
        predictions: Prediction values
        mode: Evaluation mode (train/val/test)
    """
    metrics: Dict[str, float]
    series: Dict[str, pl.Series] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    mode: str = ""


@dataclass
class StateDelta:
    """
    State delta - for updating RollingState
    
    Attributes:
        importance_vector: Feature importance vector
        feature_names_hash: Feature name hash
        best_params: Best hyperparameters
        best_objective: Best objective value
    """
    importance_vector: Optional[np.ndarray] = None
    feature_names_hash: str = ""
    best_params: Optional[Dict[str, Any]] = None
    best_objective: Optional[float] = None


@dataclass
class MethodOutput:
    """
    Complete output of Method
    
    Attributes:
        best_params: Best hyperparameters (from tuner)
        metrics_eval: Evaluation phase metrics {mode: EvalResult}
        importance_vector: Feature importance vector (aligned with current feature_names)
        feature_names_hash: Feature name hash (for error prevention)
        tune_result: Search result
        fit_result: Training result
        transform_stats: Transform statistics
        state_update: State update delta
        model_artifacts: Model artifact paths
    """
    best_params: Dict[str, Any]
    metrics_eval: Dict[str, EvalResult]
    importance_vector: np.ndarray
    feature_names_hash: str
    tune_result: Optional[TuneResult] = None
    fit_result: Optional[FitResult] = None
    transform_stats: Optional[TransformStats] = None
    state_delta: Optional[StateDelta] = None
    model_artifacts: Optional[Dict[str, Path]] = None
    
    @property
    def metrics_search(self) -> Optional[Dict[str, float]]:
        """Legacy API compatibility"""
        if self.tune_result is None:
            return None
        return {"best_value": self.tune_result.best_value}
    
    def get_state_delta(self) -> StateDelta:
        """Get delta for updating RollingState"""
        if self.state_delta is not None:
            return self.state_delta
        return StateDelta(
            importance_vector=self.importance_vector,
            feature_names_hash=self.feature_names_hash,
            best_params=self.best_params,
            best_objective=self.tune_result.best_value if self.tune_result else None,
        )
