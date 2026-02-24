"""
Split execution orchestration for one split task.
"""

from __future__ import annotations

import math
import traceback
from typing import Dict, Optional, Tuple

from ..data.data_dataclasses import GlobalStore, SplitViews
from ..method.method_factory import MethodFactory
from .experiment_dataclasses import RollingState, SplitResult, SplitTask
from .run_context import RunContext


class SplitRunner:
    """Run a single split end-to-end in a deterministic workflow."""

    def run(
        self,
        task: SplitTask,
        global_store: GlobalStore,
        rolling_state: Optional[RollingState],
        ctx: RunContext,
    ) -> SplitResult:
        split_id = task.split_id

        try:
            split_views, skip_reason = self._build_split_views(task, global_store)
            if split_views is None:
                return SplitResult(
                    split_id=split_id,
                    metrics={},
                    skipped=True,
                    skip_reason=skip_reason,
                )

            method, resolved_train_config = self._build_method(ctx, split_id)

            method_output = method.run(
                views=split_views,
                config=resolved_train_config,
                do_tune=ctx.do_tune and (method.tuner is not None),
                save_dir=ctx.split_dir(split_id),
                rolling_state=rolling_state,
            )

            metrics_flat = self._flatten_metrics(method_output.metrics_eval)
            metric_series_rows = self._flatten_metric_series(method_output.metrics_eval)
            test_predictions = None
            if "test" in method_output.metrics_eval:
                test_predictions = method_output.metrics_eval["test"].predictions

            state_delta = method_output.get_state_delta()
            return SplitResult(
                split_id=split_id,
                importance_vector=state_delta.importance_vector,
                feature_names_hash=state_delta.feature_names_hash,
                metrics=metrics_flat,
                best_params=state_delta.best_params,
                best_objective=state_delta.best_objective,
                state_delta=state_delta,
                test_predictions=test_predictions,
                test_keys=split_views.test.keys,
                test_extra=split_views.test.extra,
                metric_series_rows=metric_series_rows,
                failed=False,
            )

        except Exception as exc:
            trace_text = traceback.format_exc()
            trace_path = ctx.split_dir(split_id) / "error_traceback.txt"
            trace_path.write_text(trace_text, encoding="utf-8")
            return SplitResult(
                split_id=split_id,
                metrics={"error": f"{type(exc).__name__}: {exc}"},
                failed=True,
                skipped=False,
                skip_reason=None,
                error_message=f"{type(exc).__name__}: {exc}",
                error_trace_path=str(trace_path),
            )

    def _build_split_views(
        self,
        task: SplitTask,
        global_store: GlobalStore,
    ) -> Tuple[Optional[SplitViews], Optional[str]]:
        spec = task.splitspec

        idx_train = global_store.get_indices_by_dates(spec.train_date_list)
        idx_val = global_store.get_indices_by_dates(spec.val_date_list)
        idx_test = global_store.get_indices_by_dates(spec.test_date_list)

        empty_sets = []
        if len(idx_train) == 0:
            empty_sets.append("train")
        if len(idx_val) == 0:
            empty_sets.append("val")
        if len(idx_test) == 0:
            empty_sets.append("test")

        if empty_sets:
            return None, f"Empty sets: {', '.join(empty_sets)}"

        views = SplitViews(
            train=global_store.take(idx_train),
            val=global_store.take(idx_val),
            test=global_store.take(idx_test),
            split_spec=spec,
        )
        return views, None

    def _build_method(self, ctx: RunContext, split_id: int):
        return MethodFactory.build(
            method_config=ctx.method_config,
            train_config=ctx.train_config,
            n_trials=ctx.n_trials,
            parallel_trials=ctx.parallel_trials,
            use_ray_tune=ctx.use_ray_tune,
            base_seed=ctx.seed,
            split_id=split_id,
        )

    def _flatten_metrics(self, metrics_eval: Dict[str, object]) -> Dict[str, float]:
        metrics_flat: Dict[str, float] = {}

        for mode, eval_result in metrics_eval.items():
            for name, value in eval_result.metrics.items():
                metrics_flat[f"{mode}_{name}"] = value

        return metrics_flat

    @staticmethod
    def _coerce_date_int(raw_value: object) -> Optional[int]:
        if raw_value is None:
            return None
        try:
            date_int = int(raw_value)
        except (TypeError, ValueError):
            return None
        text = str(date_int)
        if len(text) != 8:
            return None
        return date_int

    def _flatten_metric_series(self, metrics_eval: Dict[str, object]) -> list[dict]:
        rows: list[dict] = []

        for mode, eval_result in metrics_eval.items():
            series_map = getattr(eval_result, "series", None)
            if not series_map:
                continue

            for metric_name, value_series in series_map.items():
                if metric_name.endswith("_date"):
                    continue
                date_key = f"{metric_name}_date"
                date_series = series_map.get(date_key)
                if date_series is None:
                    continue

                value_list = value_series.to_list() if hasattr(value_series, "to_list") else list(value_series)
                date_list = date_series.to_list() if hasattr(date_series, "to_list") else list(date_series)
                n = min(len(value_list), len(date_list))

                for idx in range(n):
                    date_int = self._coerce_date_int(date_list[idx])
                    if date_int is None:
                        continue
                    try:
                        value = float(value_list[idx])
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value):
                        continue
                    rows.append(
                        {
                            "mode": mode,
                            "metric": metric_name,
                            "date": date_int,
                            "value": value,
                        }
                    )

        return rows
