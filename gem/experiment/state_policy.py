"""
State policy implementations and convenience helpers.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .experiment_dataclasses import (
    FeatureImportanceState,
    RollingState,
    SampleWeightState,
    SplitResult,
    StatePolicyConfig,
    TuningState,
)


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
