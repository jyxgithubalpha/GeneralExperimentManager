"""
Experiment manager: split planning, execution and reporting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import polars as pl

from ..data.data_dataclasses import GlobalStore, SplitSpec
from ..data.data_module import DataModule
from ..data.split_generator import SplitGenerator
from ..method.method_dataclasses import TrainConfig
from .executor import BaseExecutor, LocalExecutor, RayExecutor
from .experiment_dataclasses import ExperimentConfig, RollingState, SplitResult, SplitTask
from .run_context import RunContext
from .task_dag import DynamicTaskDAG

log = logging.getLogger(__name__)
_EPS = 1e-8
_REL_IMPROVE_DENOM_FLOOR = 0.01
_ICIR_MIN_PERIODS = 20


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
        self._generate_report(output_dir)
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

    @staticmethod
    def _empty_daily_metric_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "date": pl.Int64,
                "mode": pl.Utf8,
                "metric": pl.Utf8,
                "value": pl.Float64,
                "n_split": pl.Int64,
                "is_derived": pl.Boolean,
                "source_metric": pl.Utf8,
            }
        )

    @staticmethod
    def _normalize_date_int(raw_value: object) -> Optional[int]:
        if raw_value is None:
            return None
        try:
            date_int = int(raw_value)
        except (TypeError, ValueError):
            return None
        text = str(date_int)
        if len(text) != 8:
            return None
        try:
            datetime.strptime(text, "%Y%m%d")
        except ValueError:
            return None
        return date_int

    @staticmethod
    def _aggregate_daily_metric_series(series_rows: List[Dict[str, object]]) -> pl.DataFrame:
        if not series_rows:
            return ExperimentManager._empty_daily_metric_df()

        raw = pl.DataFrame(series_rows)
        required = {"mode", "metric", "date", "value"}
        if not required.issubset(set(raw.columns)):
            missing = sorted(required - set(raw.columns))
            log.warning("Skip metric aggregation because required columns are missing: %s", missing)
            return ExperimentManager._empty_daily_metric_df()

        has_split_id = "split_id" in raw.columns
        selected_cols = [col for col in ("split_id", "mode", "metric", "date", "value") if col in raw.columns]
        df = raw.select(selected_cols).with_columns(
            pl.col("mode").cast(pl.Utf8, strict=False),
            pl.col("metric").cast(pl.Utf8, strict=False),
            pl.col("date")
            .map_elements(ExperimentManager._normalize_date_int, return_dtype=pl.Int64)
            .alias("date"),
            pl.col("value").cast(pl.Float64, strict=False),
        )
        if has_split_id:
            df = df.with_columns(pl.col("split_id").cast(pl.Int64, strict=False))

        drop_cols = ["mode", "metric", "date", "value"] + (["split_id"] if has_split_id else [])
        pre_drop_count = df.height
        df = df.drop_nulls(drop_cols).filter(pl.col("value").is_finite())
        filtered_count = pre_drop_count - df.height
        if filtered_count > 0:
            log.warning("Dropped %d invalid metric series rows before aggregation.", filtered_count)

        if df.is_empty():
            return ExperimentManager._empty_daily_metric_df()

        if has_split_id:
            grouped = (
                df.group_by(["date", "mode", "metric"])
                .agg(
                    pl.col("value").mean().alias("value"),
                    pl.col("split_id").n_unique().alias("n_split"),
                )
            )
        else:
            grouped = (
                df.group_by(["date", "mode", "metric"])
                .agg(
                    pl.col("value").mean().alias("value"),
                    pl.len().alias("n_split"),
                )
            )

        return (
            grouped.with_columns(
                pl.lit(False).alias("is_derived"),
                pl.lit(None, dtype=pl.Utf8).alias("source_metric"),
            )
            .select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
            .sort(["metric", "mode", "date"])
        )

    @staticmethod
    def _append_derived_daily_metrics(daily_df: pl.DataFrame) -> pl.DataFrame:
        if daily_df.is_empty():
            return daily_df

        rel_floor_hit_count = 0
        icir_min_periods_null_count = 0

        base = (
            daily_df.with_columns(
                pl.col("is_derived").cast(pl.Boolean, strict=False).fill_null(False),
                pl.col("source_metric").cast(pl.Utf8, strict=False),
            )
            .select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
            .sort(["metric", "mode", "date"])
        )

        frames: List[pl.DataFrame] = [base]
        existing_metrics = set(base["metric"].unique().to_list())

        top_ret_df = base.filter(pl.col("metric") == "daily_top_ret").sort(["mode", "date"])
        if not top_ret_df.is_empty() and "daily_top_ret_std" not in existing_metrics:
            top_ret_std_parts: List[pl.DataFrame] = []
            for mode in top_ret_df["mode"].unique().to_list():
                mode_df = top_ret_df.filter(pl.col("mode") == mode).sort("date")
                if mode_df.is_empty():
                    continue
                vals = mode_df["value"].to_numpy()
                n = vals.shape[0]
                if n == 0:
                    continue
                cumsum = np.cumsum(vals)
                cumsq = np.cumsum(vals * vals)
                counts = np.arange(1, n + 1, dtype=np.float64)
                means = cumsum / counts
                var = np.maximum(cumsq / counts - means * means, 0.0)
                std = np.sqrt(var)
                part = mode_df.with_columns(
                    pl.Series("value", std),
                    pl.lit("daily_top_ret_std").alias("metric"),
                    pl.lit(True).alias("is_derived"),
                    pl.lit("daily_top_ret").alias("source_metric"),
                ).select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
                top_ret_std_parts.append(part)
            if top_ret_std_parts:
                frames.append(pl.concat(top_ret_std_parts, how="vertical"))

        if "daily_top_ret_relative_improve_pct" not in existing_metrics:
            model_df = top_ret_df.rename({"value": "model_value", "n_split": "model_n_split"})
            bench_df = base.filter(pl.col("metric") == "daily_benchmark_top_ret").rename(
                {"value": "bench_value", "n_split": "bench_n_split"}
            )
            if not model_df.is_empty() and not bench_df.is_empty():
                joined = model_df.join(
                    bench_df,
                    on=["date", "mode"],
                    how="inner",
                )
                rel_floor_hit_count += joined.filter(
                    pl.col("bench_value").abs() < _REL_IMPROVE_DENOM_FLOOR
                ).height
                rel_df = (
                    joined.with_columns(
                        pl.max_horizontal(
                            pl.col("bench_value").abs(),
                            pl.lit(_REL_IMPROVE_DENOM_FLOOR),
                        ).alias("denom"),
                    )
                    .with_columns(
                        ((pl.col("model_value") - pl.col("bench_value")) / pl.col("denom") * 100.0).alias("value"),
                        pl.max_horizontal("model_n_split", "bench_n_split").alias("n_split"),
                        pl.lit("daily_top_ret_relative_improve_pct").alias("metric"),
                        pl.lit(True).alias("is_derived"),
                        pl.lit("daily_top_ret,daily_benchmark_top_ret").alias("source_metric"),
                    )
                    .drop("denom")
                    .select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
                )
                if not rel_df.is_empty():
                    frames.append(rel_df)

        if "daily_icir_expanding" not in existing_metrics:
            ic_df = base.filter(pl.col("metric") == "daily_ic").sort(["mode", "date"])
            if not ic_df.is_empty():
                icir_parts: List[pl.DataFrame] = []
                for mode in ic_df["mode"].unique().to_list():
                    mode_df = ic_df.filter(pl.col("mode") == mode).sort("date")
                    if mode_df.is_empty():
                        continue
                    vals = mode_df["value"].to_numpy()
                    n = vals.shape[0]
                    if n == 0:
                        continue
                    cumsum = np.cumsum(vals)
                    cumsq = np.cumsum(vals * vals)
                    counts = np.arange(1, n + 1, dtype=np.float64)
                    means = cumsum / counts
                    var = np.maximum(cumsq / counts - means * means, 0.0)
                    std = np.sqrt(var)
                    icir: List[Optional[float]] = []
                    for idx in range(n):
                        if counts[idx] < _ICIR_MIN_PERIODS:
                            icir_min_periods_null_count += 1
                            icir.append(None)
                        elif std[idx] > _EPS:
                            icir.append(float(means[idx] / (std[idx] + _EPS)))
                        else:
                            icir.append(None)
                    part = mode_df.with_columns(
                        pl.Series("value", icir),
                        pl.lit("daily_icir_expanding").alias("metric"),
                        pl.lit(True).alias("is_derived"),
                        pl.lit("daily_ic").alias("source_metric"),
                    ).select(["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"])
                    icir_parts.append(part)
                if icir_parts:
                    frames.append(pl.concat(icir_parts, how="vertical"))

        if rel_floor_hit_count > 0:
            log.warning(
                "Applied denominator floor %.4f for daily_top_ret_relative_improve_pct on %d rows.",
                _REL_IMPROVE_DENOM_FLOOR,
                rel_floor_hit_count,
            )
        if icir_min_periods_null_count > 0:
            log.info(
                "Set daily_icir_expanding to null for %d rows due to min_periods=%d.",
                icir_min_periods_null_count,
                _ICIR_MIN_PERIODS,
            )

        return pl.concat(frames, how="vertical").sort(["metric", "mode", "date"])

    def _generate_report(self, output_dir: Path) -> None:
        rows = []
        series_rows = []
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
            if result.metric_series_rows:
                for item in result.metric_series_rows:
                    series_rows.append(
                        {
                            "split_id": split_id,
                            "mode": item.get("mode"),
                            "metric": item.get("metric"),
                            "date": item.get("date"),
                            "value": item.get("value"),
                        }
                    )
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

        if series_rows:
            series_df = self._append_derived_daily_metrics(
                self._aggregate_daily_metric_series(series_rows)
            )
            series_csv_path = output_dir / "daily_metric_series.csv"
            series_df.write_csv(series_csv_path)
            log.info("  - Saved: %s", series_csv_path)

        config_path = output_dir / "config.json"
        config_dict = {
            "name": self.experiment_config.name,
            "seed": self.experiment_config.seed,
            "n_trials": self.experiment_config.n_trials,
            "parallel_trials": self.experiment_config.parallel_trials,
            "use_ray_tune": self.experiment_config.use_ray_tune,
            "state_policy_mode": self.experiment_config.state_policy.mode,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)
        log.info("  - Saved: %s", config_path)

        n_skipped = sum(1 for item in self._results.values() if item.skipped)
        n_failed = sum(1 for item in self._results.values() if item.failed)
        n_success = len(self._results) - n_skipped - n_failed
        log.info("  Splits: %d success, %d skipped, %d failed", n_success, n_skipped, n_failed)
