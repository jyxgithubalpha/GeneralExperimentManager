"""
LightGBM evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ...data.data_dataclasses import ProcessedViews, SplitView
from ..base import BaseEvaluator
from ..method_dataclasses import EvalResult


class LightGBMEvaluator(BaseEvaluator):
    def __init__(self, metric_names: Optional[List[str]] = None):
        self.metric_names = metric_names or ["pearsonr_ic", "pearsonr_icir"]

    def evaluate(
        self,
        model: Any,
        views: "ProcessedViews",
        modes: Optional[List[str]] = None,
    ) -> Dict[str, EvalResult]:
        from ...utils.metrics import MetricRegistry

        selected_modes = modes or ["train", "val", "test"]
        results: Dict[str, EvalResult] = {}

        for mode in selected_modes:
            view = views.get(mode)
            predictions = np.asarray(model.predict(view.X)).ravel()

            metrics: Dict[str, float] = {}
            for metric_name in self.metric_names:
                metric = MetricRegistry.get(metric_name)
                metrics[metric_name] = metric.compute(predictions, view)

            results[mode] = EvalResult(
                metrics=metrics,
                series=self._compute_series(predictions, view),
                predictions=predictions,
                mode=mode,
            )

        return results

    def _compute_series(self, pred: np.ndarray, view: SplitView) -> Dict[str, pl.Series]:
        from scipy import stats

        if view.keys is None or "date" not in view.keys.columns:
            return {"daily_ic": pl.Series("daily_ic", [], dtype=pl.Float64)}

        pred = np.asarray(pred).ravel()
        y_true = np.asarray(view.y).ravel()
        dates = view.keys["date"].to_numpy()

        daily_ic_values: List[float] = []
        daily_ic_dates: List[int] = []

        for day in np.unique(dates):
            mask = dates == day
            if int(mask.sum()) < 2:
                continue

            pred_day = pred[mask]
            true_day = y_true[mask]
            if np.std(pred_day) < 1e-8 or np.std(true_day) < 1e-8:
                continue

            ic, _ = stats.pearsonr(pred_day, true_day)
            if np.isfinite(ic):
                daily_ic_dates.append(int(day))
                daily_ic_values.append(float(ic))

        return {
            "daily_ic": pl.Series("daily_ic", daily_ic_values),
            "daily_ic_date": pl.Series("daily_ic_date", daily_ic_dates),
        }
