"""
Experiment manager: split planning, execution and reporting.
"""


import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..data.data_dataclasses import GlobalStore, SplitSpec
from ..data.data_module import DataModule
from ..data.split_generator import SplitGenerator
from ..method.method_dataclasses import TrainConfig
from .configs import ExperimentConfig
from .executor import BaseExecutor, LocalExecutor, RayExecutor
from .report import ReportGenerator
from .results import SplitResult, SplitTask
from .run_context import RunContext
from .states import RollingState
from .task_dag import DynamicTaskDAG

log = logging.getLogger(__name__)


class ExperimentManager:
    def __init__(
        self,
        split_generator: SplitGenerator,
        data_module: DataModule,
        train_config: TrainConfig,
        experiment_config: ExperimentConfig,
        method_config: Optional[Mapping[str, Any]] = None,
    ):
        self.experiment_config = experiment_config
        self.split_generator = split_generator
        self.data_module = data_module
        self.train_config = train_config
        self.method_config = method_config

        self._results: Dict[int, SplitResult] = {}
        self._global_store: Optional[GlobalStore] = None
        self._splitspec_list: Optional[List[SplitSpec]] = None

    @property
    def use_ray(self) -> bool:
        return bool(self.experiment_config.use_ray)

    @property
    def feature_names(self) -> Optional[List[str]]:
        if self._global_store is None:
            return None
        return self._global_store.feature_name_list

    @property
    def splitspec_list(self) -> Optional[List[SplitSpec]]:
        return self._splitspec_list


    def run(self) -> Dict[int, SplitResult]:
        output_dir = Path(self.experiment_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("%s", "=" * 60)
        log.info("Starting Experiment: %s", self.experiment_config.name)
        log.info("Output: %s", output_dir)
        log.info("%s", "=" * 60)

        log.info("[1/6] Generating splits...")
        gen_output = self.split_generator.generate()
        self._splitspec_list = gen_output.splitspec_list
        log.info("  - Generated %d splits", len(gen_output.splitspec_list))

        log.info("[2/6] Preparing global store...")
        self._global_store = self.data_module.prepare_global_store(
            gen_output.date_start,
            gen_output.date_end,
        )
        log.info("  - Samples: %d", self._global_store.n_samples)
        log.info("  - Features: %d", self._global_store.n_features)

        log.info("[3/6] Building tasks...")
        tasks = self._build_tasks(gen_output.splitspec_list)
        log.info("  - Built %d tasks", len(tasks))

        log.info("[4/6] Initializing state...")
        init_state = RollingState()

        mode = self.experiment_config.state_policy.mode
        log.info("[5/6] Executing with policy: %s", mode)
        results = self._execute(tasks, init_state, output_dir)
        self._results = {result.split_id: result for result in results}

        log.info("[6/6] Generating report...")
        report_generator = ReportGenerator(self.experiment_config)
        report_generator.generate(self._results, output_dir)
        log.info("%s", "=" * 60)
        log.info("Experiment completed")
        log.info("%s", "=" * 60)

        return self._results

    def _build_tasks(self, splitspec_list: List[SplitSpec]) -> List[SplitTask]:
        tasks = [
            SplitTask(
                split_id=spec.split_id,
                splitspec=spec,
                resource_request=self.experiment_config.resource_request,
            )
            for spec in splitspec_list
        ]
        return tasks

    def _build_context(self, output_dir: Path) -> RunContext:
        return RunContext(
            experiment_config=self.experiment_config,
            train_config=self.train_config,
            method_config=self.method_config,
            output_dir=output_dir,
            seed=self.experiment_config.seed,
        )

    def _create_executor(self) -> BaseExecutor:
        if not self.use_ray:
            return LocalExecutor()

        executor = RayExecutor()
        executor.init_ray(
            address=self.experiment_config.ray_address,
            num_cpus=self.experiment_config.num_cpus,
            num_gpus=self.experiment_config.num_gpus,
        )
        return executor

    def _execute(
        self,
        tasks: List[SplitTask],
        init_state: RollingState,
        output_dir: Path,
    ) -> List[SplitResult]:
        executor = self._create_executor()
        try:
            return self._run_with_executor(executor, tasks, init_state, output_dir)
        finally:
            if self.use_ray and isinstance(executor, RayExecutor):
                executor.shutdown()

    def _run_with_executor(
        self,
        executor: BaseExecutor,
        tasks: List[SplitTask],
        init_state: RollingState,
        output_dir: Path,
    ) -> List[SplitResult]:
        mode = self.experiment_config.state_policy.mode
        policy_config = self.experiment_config.state_policy

        if mode not in {"none", "per_split", "bucket"}:
            raise ValueError(
                f"Unsupported state policy mode '{mode}'. "
                "Expected one of: none, per_split, bucket."
            )

        ctx = self._build_context(output_dir)

        task_map = {task.split_id: task for task in tasks}
        dag = DynamicTaskDAG(mode=mode, policy_config=policy_config)
        execution_plan = dag.build_execution_plan(
            [task.splitspec for task in tasks],
            bucket_fn=policy_config.bucket_fn,
        )

        global_ref = executor.put(self._global_store)
        state_ref = executor.put(init_state if mode != "none" else None)

        self._print_schedule_overview(mode, execution_plan)

        submission = dag.submit(
            executor,
            execution_plan,
            task_map,
            global_ref,
            state_ref,
            ctx,
        )

        result_refs = submission.result_refs_in_order
        results = executor.get(result_refs) if self.use_ray else result_refs

        for split_id, result in zip(submission.split_ids_in_order, results):
            log.info("    Completed split %s", split_id)
            self._print_split_result(result)

        return results

    def _print_schedule_overview(self, mode: str, execution_plan) -> None:
        if mode == "none":
            log.info("  - DAG mode: NONE (all splits are independent nodes)")
            if execution_plan:
                log.info("    Parallel split count: %d", len(execution_plan[0]))
            return
        if mode == "per_split":
            log.info("  - DAG mode: PER_SPLIT (state chain)")
            log.info("    Serial stages: %d", len(execution_plan))
            return
        if mode == "bucket":
            log.info("  - DAG mode: BUCKET (parallel-in-bucket + serial-across-buckets)")
            for idx, bucket in enumerate(execution_plan, start=1):
                log.info("    Bucket %d/%d -> %d splits", idx, len(execution_plan), len(bucket))
            return
        log.info("  - DAG mode: %s", mode)

    def _print_split_result(self, result: SplitResult) -> None:
        if result.skipped:
            log.info("      [SKIPPED] %s", result.skip_reason)
            return
        if result.failed:
            error_msg = result.error_message or "Unknown error"
            log.error("      [FAILED] %s", error_msg[:160])
            return

        if not result.metrics:
            log.info("      [OK] No metrics")
            return

        if "error" in result.metrics:
            error_msg = str(result.metrics["error"])[:160]
            log.error("      [ERROR] %s", error_msg)
            return

        items = list(result.metrics.items())[:3]
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in items
        )
        log.info("      [OK] %s", metrics_str)
