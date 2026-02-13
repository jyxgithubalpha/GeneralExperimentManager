"""
Experiment manager: split planning, execution and reporting.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
from hydra.core.hydra_config import HydraConfig

from ..data.data_dataclasses import GlobalStore, SplitSpec
from ..data.data_module import DataModule
from ..data.split_generator import SplitGenerator
from ..method.method_dataclasses import TrainConfig
from .bucketing import BucketManager
from .executor import BaseExecutor, LocalExecutor, RayExecutor
from .experiment_dataclasses import ExperimentConfig, RollingState, SplitResult, SplitTask
from .run_context import RunContext
from .task_dag import DynamicTaskDAG

log = logging.getLogger(__name__)


class ExperimentManager:
    def __init__(
        self,
        split_generator: SplitGenerator,
        data_module: DataModule,
        train_config: TrainConfig,
        experiment_config: ExperimentConfig,
        method_config: Optional[Dict] = None,
        transform_pipeline_config: Optional[Dict] = None,
        adapter_config: Optional[Dict] = None,
    ):
        self.experiment_config = experiment_config
        self.split_generator = split_generator
        self.data_module = data_module
        self.train_config = train_config
        self.method_config = method_config
        self.transform_pipeline_config = transform_pipeline_config
        self.adapter_config = adapter_config

        self._results: Dict[int, SplitResult] = {}
        self._global_store: Optional[GlobalStore] = None
        self._splitspec_list: Optional[List[SplitSpec]] = None
        self._start_time: Optional[float] = None

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

    def _resolve_output_dir(self) -> Path:
        try:
            return Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            return Path(self.experiment_config.output_dir)

    def run(self) -> Dict[int, SplitResult]:
        self._start_time = time.time()
        output_dir = self._resolve_output_dir()
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
        self._generate_report(output_dir)

        elapsed = time.time() - self._start_time
        log.info("%s", "=" * 60)
        log.info("Experiment completed in %.1fs", elapsed)
        log.info("%s", "=" * 60)

        return self._results

    def _build_tasks(self, splitspec_list: List[SplitSpec]) -> List[SplitTask]:
        tasks = [
            SplitTask(
                split_id=spec.split_id,
                splitspec=spec,
                seed=self.experiment_config.seed + idx,
                resource_request=self.experiment_config.resource_request,
                train_config=self.train_config,
            )
            for idx, spec in enumerate(splitspec_list)
        ]
        return sorted(
            tasks,
            key=lambda task: (
                task.splitspec.test_date_list[0] if task.splitspec.test_date_list else 0,
                task.split_id,
            ),
        )

    def _build_context(self, output_dir: Path) -> RunContext:
        return RunContext(
            experiment_config=self.experiment_config,
            train_config=self.train_config,
            method_config=self.method_config,
            transform_config=self.transform_pipeline_config,
            adapter_config=self.adapter_config,
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

        bucket_manager = BucketManager(bucket_fn=policy_config.bucket_fn)
        execution_plan = bucket_manager.create_execution_plan(
            [task.splitspec for task in tasks],
            mode,
        )

        global_ref = executor.put(self._global_store)
        state_ref = executor.put(init_state if mode != "none" else None)

        self._print_schedule_overview(mode, execution_plan)
        dag = DynamicTaskDAG(mode=mode, policy_config=policy_config)
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

    def _generate_report(self, output_dir: Path) -> None:
        rows = []
        for split_id, result in sorted(self._results.items()):
            status = "failed" if result.failed else ("skipped" if result.skipped else "success")
            row = {
                "split_id": split_id,
                "status": status,
                "skipped": result.skipped,
                "failed": result.failed,
                "skip_reason": result.skip_reason,
                "error_message": result.error_message,
                "error_trace_path": result.error_trace_path,
            }
            if result.metrics:
                row.update(result.metrics)
            rows.append(row)

        df = pl.DataFrame(rows) if rows else pl.DataFrame(
            schema={
                "split_id": pl.Int64,
                "status": pl.Utf8,
                "skipped": pl.Boolean,
                "failed": pl.Boolean,
                "skip_reason": pl.Utf8,
                "error_message": pl.Utf8,
                "error_trace_path": pl.Utf8,
            }
        )

        csv_path = output_dir / "results_summary.csv"
        df.write_csv(csv_path)
        log.info("  - Saved: %s", csv_path)

        config_path = output_dir / "config.json"
        config_dict = {
            "name": self.experiment_config.name,
            "seed": self.experiment_config.seed,
            "n_trials": self.experiment_config.n_trials,
            "parallel_trials": self.experiment_config.parallel_trials,
            "use_ray_tune": self.experiment_config.use_ray_tune,
            "state_policy_mode": self.experiment_config.state_policy.mode,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - (self._start_time or time.time()),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)
        log.info("  - Saved: %s", config_path)

        n_skipped = sum(1 for item in self._results.values() if item.skipped)
        n_failed = sum(1 for item in self._results.values() if item.failed)
        n_success = len(self._results) - n_skipped - n_failed
        log.info("  Splits: %d success, %d skipped, %d failed", n_success, n_skipped, n_failed)
