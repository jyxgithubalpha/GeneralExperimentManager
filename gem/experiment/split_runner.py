"""
Split execution orchestration for one split task.
"""

from __future__ import annotations

import traceback
from typing import Dict, List, Optional, Tuple

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

            method = self._build_method(ctx, split_id)

            method_output = method.run(
                views=split_views,
                config=ctx.train_config,
                do_tune=ctx.do_tune and (method.tuner is not None),
                save_dir=ctx.split_dir(split_id),
                rolling_state=rolling_state,
            )

            metrics_flat = self._flatten_metrics(method_output.metrics_eval)
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
            trial_timeout=ctx.trial_timeout,
            parallel_trials=ctx.parallel_trials,
            use_ray_tune=ctx.use_ray_tune,
            base_seed=ctx.seed,
            split_id=split_id,
            adapter_config=ctx.adapter_config,
            transform_config=ctx.transform_config,
        )

    def _flatten_metrics(self, metrics_eval: Dict[str, object]) -> Dict[str, float]:
        metrics_flat: Dict[str, float] = {}

        for mode, eval_result in metrics_eval.items():
            for name, value in eval_result.metrics.items():
                metrics_flat[f"{mode}_{name}"] = value

        return metrics_flat
