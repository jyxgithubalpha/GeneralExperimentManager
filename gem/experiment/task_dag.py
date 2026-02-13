"""
Dynamic task DAG builder based on state policy mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Callable

from .executor import BaseExecutor
from .experiment_dataclasses import SplitTask, StatePolicyConfig
from ..data.data_dataclasses import SplitSpec
from .run_context import RunContext


@dataclass(frozen=True)
class DagSubmission:
    """Submission artifacts returned after DAG scheduling."""

    split_ids_in_order: List[int]
    result_refs_in_order: List[Any]
    final_state_ref: Any


class DynamicTaskDAG:
    """
    Build and submit split tasks as a dynamic DAG.

    Modes:
    - none: all splits are independent
    - per_split: each split depends on state updated by previous split
    - bucket: splits in a bucket are parallel, bucket state updates are serial
    """

    def __init__(self, mode: str, policy_config: StatePolicyConfig):
        self.mode = mode
        self.policy_config = policy_config

    def submit(
        self,
        executor: BaseExecutor,
        execution_plan: Sequence[Sequence[SplitSpec]],
        task_map: Dict[int, SplitTask],
        global_ref: Any,
        init_state_ref: Any,
        ctx: RunContext,
    ) -> DagSubmission:
        if self.mode == "none":
            def per_task_update_fn(state_ref: Any, _result_ref: Any) -> Any:
                return state_ref

            def post_batch_update_fn(state_ref: Any, _bucket_refs: List[Any]) -> Any:
                return state_ref

            return self._submit_batches(
                executor,
                execution_plan,
                task_map,
                global_ref,
                init_state_ref,
                ctx,
                per_task_update_fn,
                post_batch_update_fn,
            )
        if self.mode == "per_split":
            def per_task_update_fn(state_ref: Any, result_ref: Any) -> Any:
                return executor.submit_update_state(
                    state_ref,
                    result_ref,
                    self.policy_config,
                )

            def post_batch_update_fn(state_ref: Any, _bucket_refs: List[Any]) -> Any:
                return state_ref

            return self._submit_batches(
                executor,
                execution_plan,
                task_map,
                global_ref,
                init_state_ref,
                ctx,
                per_task_update_fn,
                post_batch_update_fn,
            )
        if self.mode == "bucket":
            def per_task_update_fn(state_ref: Any, _result_ref: Any) -> Any:
                return state_ref

            def post_batch_update_fn(state_ref: Any, bucket_refs: List[Any]) -> Any:
                return executor.submit_update_state_from_bucket(
                    state_ref,
                    bucket_refs,
                    self.policy_config,
                )

            return self._submit_batches(
                executor,
                execution_plan,
                task_map,
                global_ref,
                init_state_ref,
                ctx,
                per_task_update_fn,
                post_batch_update_fn,
            )
        raise ValueError(
            f"Unsupported DAG mode '{self.mode}'. Expected one of: none, per_split, bucket."
        )

    def _submit_batches(
        self,
        executor: BaseExecutor,
        execution_plan: Sequence[Sequence[SplitSpec]],
        task_map: Dict[int, SplitTask],
        global_ref: Any,
        state_ref: Any,
        ctx: RunContext,
        per_task_update_fn: Callable[[Any, Any], Any],
        post_batch_update_fn: Callable[[Any, List[Any]], Any],
    ) -> DagSubmission:
        split_ids: List[int] = []
        result_refs: List[Any] = []
        for batch in execution_plan:
            bucket_refs: List[Any] = []
            for spec in batch:
                task = task_map[spec.split_id]
                result_ref = executor.submit_run_split(task, global_ref, state_ref, ctx)
                bucket_refs.append(result_ref)
                split_ids.append(task.split_id)
                result_refs.append(result_ref)
                state_ref = per_task_update_fn(state_ref, result_ref)
            state_ref = post_batch_update_fn(state_ref, bucket_refs)
        return DagSubmission(split_ids, result_refs, state_ref)
