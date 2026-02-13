"""
Data structure definitions for state management

Architecture design:
- BaseState: State base class, all pluggable states inherit from this class
- RollingState: State container, manages multiple BaseState instances
- Specific state classes: FeatureImportanceState, TuningState, DataWeightState, etc.
"""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import polars as pl

from ..data.data_dataclasses import SplitSpec, SplitView
from ..method.method_dataclasses import StateDelta, TrainConfig
from ..utils.hash_utils import hash_feature_names


# =============================================================================
# State base class
# =============================================================================


class BaseState(ABC):
    """
    Pluggable state base class
    
    All state components that can be stored in RollingState should inherit from this class.
    Each State class should define its own state_key as a unique identifier.
    """
    
    @classmethod
    @abstractmethod
    def state_key(cls) -> str:
        """Unique identifier for the state"""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update state"""
        pass
    
    def to_transform_context(self) -> Dict[str, Any]:
        """
        Convert to context usable by transform
        
        The returned dict will be passed to Transform.fit/transform methods
        """
        return {}


@dataclass
class StatePolicyConfig:
    """
    State policy configuration    
    Attributes:
        mode: Policy mode
        bucket_fn: Bucket grouping function
        ema_alpha: EMA smoothing coefficient
        importance_topk: Keep only top-k features
        normalize_importance: Whether to normalize importance    
    """
    mode: str = "none"
    bucket_fn: Optional[Callable[[SplitSpec], str]] = None
    ema_alpha: float = 0.3
    importance_topk: Optional[int] = None
    normalize_importance: bool = True


@dataclass
class ResourceRequest:
    """Resource request"""
    trial_gpus: float = 1.0  # Number of GPUs per trial
    final_train_gpus: float = 1.0  # Number of GPUs for final training
    trial_cpus: float = 1.0
    final_train_cpus: float = 1.0


@dataclass
class VisualizationConfig:
    enabled: bool = False
    export_csv: bool = True
    heatmap: bool = True
    animation: bool = True
    distribution: bool = True
    show: bool = False
    interval: int = 800
    output_subdir: str = "plots"


@dataclass
class ExperimentConfig:
    """
    Experiment configuration
    """
    name: str
    output_dir: Path
    state_policy: "StatePolicyConfig" = None  # type: ignore  # Set via Hydra
    n_trials: int = 50
    trial_timeout: Optional[int] = None
    parallel_trials: int = 1
    use_ray_tune: Optional[bool] = None
    seed: int = 42
    ray_address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    use_ray: bool = False
    resource_request: Optional[ResourceRequest] = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        if self.n_trials < 0:
            raise ValueError(f"n_trials must be >= 0, got {self.n_trials}")
        if self.parallel_trials <= 0:
            raise ValueError(
                f"parallel_trials must be > 0, got {self.parallel_trials}"
            )
        if self.state_policy is None:
            self.state_policy = StatePolicyConfig()


@dataclass
class SplitTask:
    """
    Split task - built by ExperimentManager
    Attributes:
        split_id: Split ID
        splitspec: Split specification
        seed: Random seed
        resource_request: Resource request
    """
    split_id: int
    splitspec: SplitSpec
    seed: int = 42
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    train_config: Optional["TrainConfig"] = None


@dataclass
class SplitResult:
    """
    Split result - result of a single split training    
    Attributes:
        split_id: Split identifier
        importance_vector: Feature importance vector
        feature_names_hash: Feature name hash
        industry_delta: Industry incremental weights
        metrics: Evaluation metrics
        best_params: Best hyperparameters
        best_objective: Best objective value
        skipped: Whether skipped
        skip_reason: Skip reason
        test_predictions: Test set predictions (for backtesting)
        test_keys: Test set keys (date, code)
        test_extra: Test set extra data (bench, score, etc.)
    """
    split_id: int
    importance_vector: Optional[np.ndarray] = None
    feature_names_hash: Optional[str] = None
    industry_delta: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_objective: Optional[float] = None
    state_delta: Optional[StateDelta] = None
    skipped: bool = False
    failed: bool = False
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None
    error_trace_path: Optional[str] = None
    test_predictions: Optional[np.ndarray] = None
    test_keys: Optional["pl.DataFrame"] = None
    test_extra: Optional["pl.DataFrame"] = None


# =============================================================================
# Specific state classes
# =============================================================================

@dataclass
class FeatureImportanceState(BaseState):
    """
    Feature importance state
    
    Used for storing and updating EMA of feature importance, can be passed to FeatureWeightTransform.
    """
    importance_ema: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    feature_names_hash: Optional[str] = None
    alpha: float = 0.3
    
    @classmethod
    def state_key(cls) -> str:
        return "feature_importance"
    
    def update(
        self,
        new_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """Update feature importance EMA"""
        if new_importance is None:
            return
        
        # Update alpha
        if alpha is not None:
            self.alpha = alpha
        
        # Check feature name consistency
        if feature_names is not None:
            new_hash = hash_feature_names(feature_names)
            if self.feature_names_hash is not None and self.feature_names_hash != new_hash:
                raise ValueError(f"Feature names hash mismatch: {self.feature_names_hash} vs {new_hash}")
            self.feature_names = feature_names
            self.feature_names_hash = new_hash
        
        # Update EMA
        if self.importance_ema is None:
            self.importance_ema = new_importance.copy()
        else:
            self.importance_ema = self.alpha * new_importance + (1 - self.alpha) * self.importance_ema
    
    def to_transform_context(self) -> Dict[str, Any]:
        """Pass feature_weights to FeatureWeightTransform"""
        return {
            "feature_weights": self.importance_ema,
            "feature_names": self.feature_names,
        }
    
    def get_topk_indices(self, k: int) -> np.ndarray:
        """Get indices of top-k important features"""
        if self.importance_ema is None:
            return np.array([])
        k = min(k, len(self.importance_ema))
        return np.argsort(self.importance_ema)[-k:]


@dataclass
class SampleWeightState(BaseState):
    """
    Sample weight state
    
    Used for computing training sample weights, supports weighting by asset/time/industry.
    """
    asset_weights: Optional[Dict[str, float]] = None
    industry_weights: Optional[Dict[str, float]] = None
    time_weights: Optional[Dict[int, float]] = None
    
    @classmethod
    def state_key(cls) -> str:
        return "sample_weight"
    
    def update(
        self,
        asset_weights: Optional[Dict[str, float]] = None,
        industry_weights: Optional[Dict[str, float]] = None,
        time_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """Update sample weights"""
        if asset_weights is not None:
            self.asset_weights = asset_weights
        if industry_weights is not None:
            self.industry_weights = industry_weights
        if time_weights is not None:
            self.time_weights = time_weights
    
    def to_transform_context(self) -> Dict[str, Any]:
        return {
            "asset_weights": self.asset_weights,
            "industry_weights": self.industry_weights,
            "time_weights": self.time_weights,
        }
    
    def get_sample_weight(
        self,
        keys: pl.DataFrame,
        group: Optional[pl.DataFrame] = None,
        industry_col: str = "industry",
    ) -> np.ndarray:
        """Compute sample weights"""
        n = keys.height
        weights = np.ones(n, dtype=np.float32)
        
        if self.asset_weights is not None:
            codes = keys["code"].to_numpy()
            for i, code in enumerate(codes):
                if code in self.asset_weights:
                    weights[i] *= self.asset_weights[code]
        
        if self.time_weights is not None:
            dates = keys["date"].to_numpy()
            for i, d in enumerate(dates):
                if d in self.time_weights:
                    weights[i] *= self.time_weights[d]
        
        if self.industry_weights is not None and group is not None:
            if industry_col in group.columns:
                industries = group[industry_col].to_numpy()
                for i, ind in enumerate(industries):
                    if ind in self.industry_weights:
                        weights[i] *= self.industry_weights[ind]

        return weights

    def get_weights_for_view(
        self,
        view: SplitView,
        industry_col: str = "industry",
    ) -> np.ndarray:
        return self.get_sample_weight(
            keys=view.keys,
            group=view.group if view.group is not None else view.extra,
            industry_col=industry_col,
        )


@dataclass
class TuningState(BaseState):
    """
    Tuning state - for hyperparameter search optimization
    """
    last_best_params: Optional[Dict[str, Any]] = None
    params_history: List[Dict[str, Any]] = field(default_factory=list)
    objective_history: List[float] = field(default_factory=list)
    search_space_shrink_ratio: float = 0.5
    
    @classmethod
    def state_key(cls) -> str:
        return "tuning"
    
    def update(self, best_params: Dict[str, Any], best_objective: float) -> None:
        """Update tuning state"""
        self.last_best_params = best_params.copy()
        self.params_history.append(best_params.copy())
        self.objective_history.append(best_objective)
    
    def to_transform_context(self) -> Dict[str, Any]:
        return {
            "last_best_params": self.last_best_params,
        }
    
    def get_shrunk_space(
        self,
        base_space: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Shrink search space based on history"""
        if self.last_best_params is None:
            return base_space
        
        shrunk_space = {}
        ratio = self.search_space_shrink_ratio
        
        for name, (low, high) in base_space.items():
            if name in self.last_best_params:
                center = self.last_best_params[name]
                half_range = (high - low) * ratio / 2
                new_low = max(low, center - half_range)
                new_high = min(high, center + half_range)
                shrunk_space[name] = (new_low, new_high)
            else:
                shrunk_space[name] = (low, high)
        
        return shrunk_space


# =============================================================================
# State container
# =============================================================================

@dataclass
class RollingState:
    """
    Rolling state container - manages multiple pluggable State components
    
    Supports dynamic registration and retrieval of different types of State, and can convert all State 
    into unified transform context to pass to Transform Pipeline.
    
    Attributes:
        states: State dictionary {state_key: BaseState}
        split_history: Historical split ID list
        metadata: Additional metadata
    """
    states: Dict[str, BaseState] = field(default_factory=dict)
    split_history: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def register_state(self, state: BaseState) -> None:
        """Register a state component"""
        self.states[state.state_key()] = state
    
    def get_state(self, state_class: Type[BaseState]) -> Optional[BaseState]:
        """Get state of specified type"""
        key = state_class.state_key()
        return self.states.get(key)
    
    def get_or_create_state(self, state_class: Type[BaseState]) -> BaseState:
        """Get or create state of specified type"""
        key = state_class.state_key()
        if key not in self.states:
            self.states[key] = state_class()
        return self.states[key]  # type: ignore
    
    def has_state(self, state_class: Type[BaseState]) -> bool:
        """Check if state of specified type exists"""
        return state_class.state_key() in self.states
    
    def to_transform_context(self) -> Dict[str, Any]:
        """
        Convert all states to transform context
        
        The returned dict can be passed to BaseTransformPipeline.process_views()
        """
        context = {}
        for state in self.states.values():
            context.update(state.to_transform_context())
        return context
    
    # =========================================================================
    # Convenience methods - shortcuts for common operations
    # =========================================================================
    
    def update_importance(
        self,
        new_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.3,
    ) -> None:
        """Update feature importance (convenience method)"""
        state = self.get_or_create_state(FeatureImportanceState)
        state.update(new_importance, feature_names, alpha)
    
    def update_tuning(
        self,
        best_params: Dict[str, Any],
        best_objective: float,
    ) -> None:
        """Update tuning state (convenience method)"""
        state = self.get_or_create_state(TuningState)
        state.update(best_params, best_objective)
    
    @property
    def feature_importance(self) -> Optional[FeatureImportanceState]:
        """Get feature importance state (convenience property)"""
        return self.get_state(FeatureImportanceState)
    
    @property
    def tuning(self) -> Optional[TuningState]:
        """Get tuning state (convenience property)"""
        return self.get_state(TuningState)
    
    @property
    def sample_weight(self) -> Optional[SampleWeightState]:
        """Get sample weight state (convenience property)"""
        return self.get_state(SampleWeightState)
    
    def save(self, path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path) -> "RollingState":
        with open(path, 'rb') as f:
            return pickle.load(f)
