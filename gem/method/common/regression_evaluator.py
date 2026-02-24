"""
Generic regression evaluator for models exposing a predict() method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ...data.data_dataclasses import ProcessedViews, SplitView
from ..base import BaseEvaluator
from ..method_dataclasses import EvalResult
from .portfolio_backtest import (
    PortfolioBacktestCalculator,
    PortfolioBacktestConfig,
)


class RegressionEvaluator(BaseEvaluator):
    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        portfolio_top_k: int = 500,
        portfolio_money: float = 1.5e9,
        portfolio_min_trade_money: float = 1.0,
        portfolio_ret_scale: float = 100.0,
        ret_col_candidates: Optional[List[str]] = None,
        liquidity_col_candidates: Optional[List[str]] = None,
        benchmark_col_candidates: Optional[List[str]] = None,
        benchmark_member_cols: Optional[List[str]] = None,
        use_benchmark_ensemble: bool = True,
    ):
        self.metric_names = metric_names or [
            "pearsonr_ic",
            "pearsonr_icir",
            "top_ret",
            "top_ret_std",
            "model_benchmark_corr",
            "top_ret_relative_improve_pct",
        ]
        self.portfolio_backtest = PortfolioBacktestCalculator(
            PortfolioBacktestConfig(
                top_k=portfolio_top_k,
                money=portfolio_money,
                min_trade_money=portfolio_min_trade_money,
                ret_scale=portfolio_ret_scale,
                ret_col_candidates=tuple(
                    ret_col_candidates or ("ret__ret_value", "ret_value", "ret")
                ),
                liquidity_col_candidates=tuple(
                    liquidity_col_candidates
                    or ("liquidity__liquidity_value", "liquidity_value", "liquidity")
                ),
                benchmark_col_candidates=tuple(
                    benchmark_col_candidates
                    or (
                        "score__score_value",
                        "benchmark__benchmark_value",
                        "score_value",
                        "benchmark_value",
                    )
                ),
                benchmark_member_cols=tuple(
                    benchmark_member_cols
                    or (
                        "bench1__bench1_value",
                        "bench2__bench2_value",
                        "bench3__bench3_value",
                        "bench4__bench4_value",
                        "bench5__bench5_value",
                        "bench6__bench6_value",
                    )
                ),
                use_benchmark_ensemble=use_benchmark_ensemble,
            )
        )

    def evaluate(
        self,
        model: Any,
        views: "ProcessedViews",
        modes: Optional[List[str]] = None,
    ) -> Dict[str, EvalResult]:
        from ...utils.metrics import MetricRegistry

        selected_modes = modes or ["train", "val", "test"]
        results: Dict[str, EvalResult] = {}
        registry_names = set(MetricRegistry.list_available())
        backtest_names = PortfolioBacktestCalculator.METRIC_NAMES
        needs_backtest = any(name in backtest_names for name in self.metric_names)

        for mode in selected_modes:
            view = views.get(mode)
            predictions = self._predict(model, view.X)

            backtest_metrics: Dict[str, float] = {}
            backtest_series: Dict[str, pl.Series] = {}
            if needs_backtest:
                backtest_metrics, backtest_series = self.portfolio_backtest.compute(
                    predictions,
                    view,
                )

            metrics: Dict[str, float] = {}
            for metric_name in self.metric_names:
                if metric_name in registry_names:
                    metric = MetricRegistry.get(metric_name)
                    metrics[metric_name] = metric.compute(predictions, view)
                    continue
                if metric_name in backtest_names:
                    value = backtest_metrics.get(metric_name, np.nan)
                    metrics[metric_name] = float(value)
                    continue
                raise ValueError(
                    f"Metric '{metric_name}' is not supported by RegressionEvaluator."
                )

            series = self._compute_series(predictions, view)
            series.update(backtest_series)
            results[mode] = EvalResult(
                metrics=metrics,
                series=series,
                predictions=predictions,
                mode=mode,
            )

        return results

    @staticmethod
    def _predict(model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X)).ravel()
        if callable(model):
            return np.asarray(model(X)).ravel()
        raise ValueError("Model does not implement predict().")

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
