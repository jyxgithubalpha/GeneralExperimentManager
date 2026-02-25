"""
Dynamic task DAG builder based on state policy mode.
"""


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .executor import BaseExecutor
from .configs import StatePolicyConfig
from .results import SplitTask
from ..data.data_dataclasses import SplitSpec
from .run_context import RunContext


@dataclass(frozen=True)
class DagSubmission:
    """Submission artifacts returned after DAG scheduling."""

    split_ids_in_order: List[int]
    result_refs_in_order: List[Any]
    final_state_ref: Any


def _get_test_start(splitspec: SplitSpec) -> int:
    return splitspec.test_date_list[0] if splitspec.test_date_list else 0


def _default_bucket_fn(splitspec: SplitSpec) -> str:
    """Default bucket function: group by quarter."""
    test_start = _get_test_start(splitspec)
    year = test_start // 10000
    month = (test_start % 10000) // 100
    quarter = (month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def build_execution_plan(
    splitspecs: Sequence[SplitSpec],
    mode: str,
    bucket_fn: Optional[Callable[[SplitSpec], str]] = None,
) -> List[List[SplitSpec]]:
    if mode == "none":
        return [list(splitspecs)]

    sorted_specs = sorted(splitspecs, key=_get_test_start)
    if mode == "per_split":
        return [[spec] for spec in sorted_specs]

    if mode == "bucket":
        key_fn = bucket_fn or _default_bucket_fn
        buckets: Dict[str, List[SplitSpec]] = {}
        for spec in sorted_specs:
            key = key_fn(spec)
            buckets.setdefault(key, []).append(spec)

        bucket_order = sorted(
            buckets.keys(),
            key=lambda key: min(_get_test_start(spec) for spec in buckets[key]),
        )
        return [buckets[key] for key in bucket_order]

    raise ValueError(
        f"Unsupported DAG mode '{mode}'. Expected one of: none, per_split, bucket."
    )


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

    def build_execution_plan(
        self,
        splitspecs: Sequence[SplitSpec],
        bucket_fn: Optional[Callable[[SplitSpec], str]] = None,
    ) -> List[List[SplitSpec]]:
        return build_execution_plan(splitspecs, mode=self.mode, bucket_fn=bucket_fn)

    def _get_update_fns(
        self,
        executor: BaseExecutor,
    ) -> tuple[Callable[[Any, Any], Any], Callable[[Any, List[Any]], Any]]:
        """Get per-task and post-batch update functions based on mode."""
        noop_task = lambda state_ref, _: state_ref
        noop_batch = lambda state_ref, _: state_ref

        if self.mode == "none":
            return noop_task, noop_batch

        if self.mode == "per_split":
            def per_task(state_ref: Any, result_ref: Any) -> Any:
                return executor.submit_update_state(state_ref, result_ref, self.policy_config)
            return per_task, noop_batch

        if self.mode == "bucket":
            def post_batch(state_ref: Any, bucket_refs: List[Any]) -> Any:
                return executor.submit_update_state_from_bucket(state_ref, bucket_refs, self.policy_config)
            return noop_task, post_batch

        raise ValueError(f"Unsupported DAG mode '{self.mode}'. Expected one of: none, per_split, bucket.")

    def submit(
        self,
        executor: BaseExecutor,
        execution_plan: Sequence[Sequence[SplitSpec]],
        task_map: Dict[int, SplitTask],
        global_ref: Any,
        init_state_ref: Any,
        ctx: RunContext,
    ) -> DagSubmission:
        per_task_fn, post_batch_fn = self._get_update_fns(executor)

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
                state_ref = per_task_fn(state_ref, result_ref)
            state_ref = post_batch_fn(state_ref, bucket_refs)

        return DagSubmission(split_ids, result_refs, state_ref)
