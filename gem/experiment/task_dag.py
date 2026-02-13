"""
Dynamic task DAG builder based on state policy mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .executor import BaseExecutor
from .experiment_dataclasses import SplitTask, StatePolicyConfig
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
        tasks: List[SplitTask],
        execution_plan: Sequence[Sequence[Any]],
        task_map: Dict[int, SplitTask],
        global_ref: Any,
        init_state_ref: Any,
        ctx: RunContext,
    ) -> DagSubmission:
        if self.mode == "none":
            return self._submit_none(executor, tasks, global_ref, init_state_ref, ctx)
        if self.mode == "per_split":
            return self._submit_per_split(
                executor,
                execution_plan,
                task_map,
                global_ref,
                init_state_ref,
                ctx,
            )
        if self.mode == "bucket":
            return self._submit_bucket(
                executor,
                execution_plan,
                task_map,
                global_ref,
                init_state_ref,
                ctx,
            )
        raise ValueError(
            f"Unsupported DAG mode '{self.mode}'. Expected one of: none, per_split, bucket."
        )

    def _submit_none(
        self,
        executor: BaseExecutor,
        tasks: List[SplitTask],
        global_ref: Any,
        state_ref: Any,
        ctx: RunContext,
    ) -> DagSubmission:
        split_ids: List[int] = []
        result_refs: List[Any] = []
        for task in tasks:
            split_ids.append(task.split_id)
            result_refs.append(executor.submit_run_split(task, global_ref, state_ref, ctx))
        return DagSubmission(split_ids, result_refs, state_ref)

    def _submit_per_split(
        self,
        executor: BaseExecutor,
        execution_plan: Sequence[Sequence[Any]],
        task_map: Dict[int, SplitTask],
        global_ref: Any,
        init_state_ref: Any,
        ctx: RunContext,
    ) -> DagSubmission:
        split_ids: List[int] = []
        result_refs: List[Any] = []

        state_ref = init_state_ref
        for batch in execution_plan:
            for spec in batch:
                task = task_map[spec.split_id]
                result_ref = executor.submit_run_split(task, global_ref, state_ref, ctx)
                state_ref = executor.submit_update_state(
                    state_ref,
                    result_ref,
                    self.policy_config,
                )
                split_ids.append(task.split_id)
                result_refs.append(result_ref)

        return DagSubmission(split_ids, result_refs, state_ref)

    def _submit_bucket(
        self,
        executor: BaseExecutor,
        execution_plan: Sequence[Sequence[Any]],
        task_map: Dict[int, SplitTask],
        global_ref: Any,
        init_state_ref: Any,
        ctx: RunContext,
    ) -> DagSubmission:
        split_ids: List[int] = []
        result_refs: List[Any] = []

        state_ref = init_state_ref
        for batch in execution_plan:
            bucket_refs: List[Any] = []

            for spec in batch:
                task = task_map[spec.split_id]
                result_ref = executor.submit_run_split(task, global_ref, state_ref, ctx)
                bucket_refs.append(result_ref)
                split_ids.append(task.split_id)
                result_refs.append(result_ref)

            state_ref = executor.submit_update_state_from_bucket(
                state_ref,
                bucket_refs,
                self.policy_config,
            )

        return DagSubmission(split_ids, result_refs, state_ref)
