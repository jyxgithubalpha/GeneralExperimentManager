"""
Execution backends for split execution and state updates.
"""


from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..data.data_dataclasses import GlobalStore
from .configs import StatePolicyConfig
from .results import SplitResult, SplitTask
from .run_context import RunContext
from .states import RollingState


class BaseExecutor(ABC):
    @abstractmethod
    def put(self, obj: Any) -> Any:
        pass

    @abstractmethod
    def submit_run_split(
        self,
        task: SplitTask,
        global_store: Any,
        state: Any,
        ctx: RunContext,
    ) -> Any:
        pass

    @abstractmethod
    def get(self, refs: Any) -> Any:
        pass

    @abstractmethod
    def submit_update_state(
        self,
        state: Any,
        result: Any,
        config: StatePolicyConfig,
    ) -> Any:
        pass

    @abstractmethod
    def submit_update_state_from_bucket(
        self,
        state: Any,
        results: List[Any],
        config: StatePolicyConfig,
    ) -> Any:
        pass


class LocalExecutor(BaseExecutor):
    """In-process execution backend."""

    def __init__(self):
        from .split_runner import SplitRunner

        self.runner = SplitRunner()

    def put(self, obj: Any) -> Any:
        return obj

    def submit_run_split(
        self,
        task: SplitTask,
        global_store: GlobalStore,
        state: Optional[RollingState],
        ctx: RunContext,
    ) -> SplitResult:
        return self.runner.run(task, global_store, state, ctx)

    def get(self, refs: Any) -> Any:
        return refs

    def submit_update_state(
        self,
        state: Optional[RollingState],
        result: SplitResult,
        config: StatePolicyConfig,
    ) -> RollingState:
        from .states import update_state

        return update_state(state, result, config)

    def submit_update_state_from_bucket(
        self,
        state: Optional[RollingState],
        results: List[SplitResult],
        config: StatePolicyConfig,
    ) -> RollingState:
        from .states import update_state_from_bucket_results

        return update_state_from_bucket_results(state, results, config)


class RayExecutor(BaseExecutor):
    """Ray-based distributed execution backend."""

    def __init__(self):
        self._ray = None
        self._remote_funcs: dict[str, Any] = {}

    def init_ray(self, address: Optional[str] = None, **kwargs) -> None:
        try:
            import ray
        except ImportError as exc:
            raise ImportError("ray is required. Install with: pip install ray") from exc

        self._ray = ray
        init_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not ray.is_initialized():
            ray.init(address=address, **init_kwargs)

        self._create_remote_funcs()

    def _create_remote_funcs(self) -> None:
        ray = self._ray

        @ray.remote
        def run_split_remote(task, global_store, state, ctx):
            from .split_runner import SplitRunner

            runner = SplitRunner()
            return runner.run(task, global_store, state, ctx)

        @ray.remote
        def update_state_remote(state, result, config):
            from .states import update_state

            return update_state(state, result, config)

        @ray.remote
        def update_state_from_bucket_remote(state, results, config):
            import ray as _ray
            from .states import update_state_from_bucket_results

            # Ray does not auto-resolve ObjectRefs nested inside a list argument.
            if results and isinstance(results[0], _ray.ObjectRef):
                results = _ray.get(results)
            return update_state_from_bucket_results(state, results, config)

        self._remote_funcs["run_split"] = run_split_remote
        self._remote_funcs["update_state"] = update_state_remote
        self._remote_funcs["update_state_from_bucket"] = update_state_from_bucket_remote

    def put(self, obj: Any) -> Any:
        return self._ray.put(obj)

    def submit_run_split(
        self,
        task: SplitTask,
        global_store_ref: Any,
        state_ref: Any,
        ctx: RunContext,
    ) -> Any:
        options = {}
        resource_request = getattr(task, "resource_request", None)
        if resource_request is not None:
            trial_gpus = getattr(resource_request, "trial_gpus", None)
            trial_cpus = getattr(resource_request, "trial_cpus", None)
            if trial_gpus is not None and float(trial_gpus) > 0:
                options["num_gpus"] = float(trial_gpus)
            if trial_cpus is not None and float(trial_cpus) > 0:
                options["num_cpus"] = float(trial_cpus)

        remote_func = self._remote_funcs["run_split"]
        if options:
            remote_func = remote_func.options(**options)
        return remote_func.remote(task, global_store_ref, state_ref, ctx)

    def get(self, refs: Any) -> Any:
        return self._ray.get(refs)

    def submit_update_state(
        self,
        state_ref: Any,
        result_ref: Any,
        config: StatePolicyConfig,
    ) -> Any:
        return self._remote_funcs["update_state"].remote(state_ref, result_ref, config)

    def submit_update_state_from_bucket(
        self,
        state_ref: Any,
        results: List[Any],
        config: StatePolicyConfig,
    ) -> Any:
        return self._remote_funcs["update_state_from_bucket"].remote(
            state_ref,
            results,
            config,
        )

    def shutdown(self) -> None:
        if self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()
