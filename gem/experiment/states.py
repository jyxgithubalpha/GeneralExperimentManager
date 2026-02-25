"""
State management: base classes, concrete states, and update policies.
"""


import copy
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import polars as pl

from ..data.data_dataclasses import SplitView
from .configs import StatePolicyConfig
from .results import SplitResult


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
        
        # Update feature names
        if feature_names is not None:
            self.feature_names = feature_names
        
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


# =============================================================================
# State policy classes
# =============================================================================

def _result_importance_vector(result: SplitResult) -> Optional[np.ndarray]:
    if result.state_delta is not None and result.state_delta.importance_vector is not None:
        return result.state_delta.importance_vector
    return result.importance_vector


def _result_feature_hash(result: SplitResult) -> str:
    if result.state_delta is not None and result.state_delta.feature_names_hash:
        return result.state_delta.feature_names_hash
    return result.feature_names_hash or ""


class StatePolicy(ABC):
    @abstractmethod
    def update_state(self, prev_state: RollingState, split_result: SplitResult) -> RollingState:
        pass

    @abstractmethod
    def aggregate_importance(self, results: List[SplitResult]) -> np.ndarray:
        pass

    @abstractmethod
    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        pass


class NoStatePolicy(StatePolicy):
    def update_state(self, prev_state: RollingState, split_result: SplitResult) -> RollingState:
        return prev_state

    def aggregate_importance(self, results: List[SplitResult]) -> np.ndarray:
        vectors = [
            _result_importance_vector(r)
            for r in results
            if _result_importance_vector(r) is not None
        ]
        if not vectors:
            return np.array([])
        return np.mean(vectors, axis=0)

    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        return prev_state


class EMAStatePolicy(StatePolicy):
    def __init__(
        self,
        alpha: float = 0.3,
        topk: Optional[int] = None,
        normalize: bool = True,
    ):
        self.alpha = alpha
        self.topk = topk
        self.normalize = normalize

    def _clone_state(self, state: RollingState) -> RollingState:
        return RollingState(
            states=copy.deepcopy(state.states),
            split_history=state.split_history.copy(),
            metadata=state.metadata.copy(),
        )

    def _post_process_importance(self, fi_state: FeatureImportanceState) -> None:
        if fi_state.importance_ema is None:
            return

        if self.topk is not None and self.topk > 0:
            mask = np.zeros_like(fi_state.importance_ema)
            topk_indices = np.argsort(fi_state.importance_ema)[-self.topk :]
            mask[topk_indices] = 1.0
            fi_state.importance_ema = fi_state.importance_ema * mask

        if self.normalize:
            total = np.sum(fi_state.importance_ema)
            if total > 0:
                fi_state.importance_ema = fi_state.importance_ema / total

    def update_state(self, prev_state: RollingState, split_result: SplitResult) -> RollingState:
        new_state = self._clone_state(prev_state)
        if split_result.skipped or split_result.failed:
            new_state.split_history.append(split_result.split_id)
            return new_state

        importance_vector = _result_importance_vector(split_result)
        if importance_vector is not None:
            new_state.update_importance(importance_vector, alpha=self.alpha)
            fi_state = new_state.get_state(FeatureImportanceState)
            if fi_state is not None:
                self._post_process_importance(fi_state)

        if split_result.industry_delta is not None:
            sw_state = new_state.get_or_create_state(SampleWeightState)
            if sw_state.industry_weights is None:
                sw_state.industry_weights = split_result.industry_delta.copy()
            else:
                for key, value in split_result.industry_delta.items():
                    if key in sw_state.industry_weights:
                        sw_state.industry_weights[key] = (
                            self.alpha * value + (1 - self.alpha) * sw_state.industry_weights[key]
                        )
                    else:
                        sw_state.industry_weights[key] = value

        new_state.split_history.append(split_result.split_id)
        return new_state

    def aggregate_importance(self, results: List[SplitResult]) -> np.ndarray:
        if not results:
            return np.array([])

        vectors = [
            _result_importance_vector(r)
            for r in results
            if (not r.failed and not r.skipped and _result_importance_vector(r) is not None)
        ]
        if not vectors:
            return np.array([])

        hashes = {_result_feature_hash(r) for r in results if _result_feature_hash(r)}
        if len(hashes) > 1:
            raise ValueError(f"Inconsistent feature_names_hash values in bucket: {hashes}")

        return np.mean(vectors, axis=0)

    def update_state_from_bucket(
        self,
        prev_state: RollingState,
        agg_importance: np.ndarray,
        feature_names_hash: str,
    ) -> RollingState:
        new_state = self._clone_state(prev_state)

        if agg_importance is not None and len(agg_importance) > 0:
            new_state.update_importance(agg_importance, alpha=self.alpha)
            fi_state = new_state.get_state(FeatureImportanceState)
            if fi_state is not None:
                self._post_process_importance(fi_state)

        return new_state


class StatePolicyFactory:
    @staticmethod
    def create(config: StatePolicyConfig) -> StatePolicy:
        if config.mode == "none":
            return NoStatePolicy()
        if config.mode in ("per_split", "bucket"):
            return EMAStatePolicy(
                alpha=config.ema_alpha,
                topk=config.importance_topk,
                normalize=config.normalize_importance,
            )
        return NoStatePolicy()


# =============================================================================
# Convenience functions
# =============================================================================

def update_state(
    prev_state: Optional[RollingState],
    split_result: SplitResult,
    config: StatePolicyConfig,
) -> RollingState:
    policy = StatePolicyFactory.create(config)
    return policy.update_state(prev_state or RollingState(), split_result)


def aggregate_bucket_results(
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> np.ndarray:
    policy = StatePolicyFactory.create(config)
    return policy.aggregate_importance(results)


def update_state_from_bucket_results(
    prev_state: Optional[RollingState],
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> RollingState:
    policy = StatePolicyFactory.create(config)
    agg_importance = policy.aggregate_importance(results)

    feature_names_hash = ""
    for result in results:
        candidate_hash = _result_feature_hash(result)
        if candidate_hash:
            feature_names_hash = candidate_hash
            break

    return policy.update_state_from_bucket(
        prev_state or RollingState(),
        agg_importance,
        feature_names_hash,
    )
