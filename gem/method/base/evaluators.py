"""
Evaluators - Evaluation components for different tasks.

Contains:
- RegressionEvaluator: Generic regression evaluator
- PortfolioBacktestCalculator: Portfolio backtest metrics
"""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from ...data.data_dataclasses import ProcessedViews, SplitView
from .base_evaluator import BaseEvaluator
from ..method_dataclasses import EvalResult


# =============================================================================
# Portfolio Backtest
# =============================================================================

@dataclass(frozen=True)
class PortfolioBacktestConfig:
    top_k: int = 500
    money: float = 1.5e9
    min_trade_money: float = 1.0
    ret_scale: float = 1.0
    relative_improve_denom_floor: float = 0.01
    relative_improve_output_scale: float = 1.0
    ret_col_candidates: Tuple[str, ...] = ("ret__ret_value", "ret_value", "ret")
    liquidity_col_candidates: Tuple[str, ...] = (
        "liquidity__liquidity_value",
        "liquidity_value",
        "liquidity",
    )
    benchmark_col_candidates: Tuple[str, ...] = (
        "score__score_value",
        "benchmark__benchmark_value",
        "score_value",
        "benchmark_value",
    )
    benchmark_member_cols: Tuple[str, ...] = (
        "bench1__bench1_value",
        "bench2__bench2_value",
        "bench3__bench3_value",
        "bench4__bench4_value",
        "bench5__bench5_value",
        "bench6__bench6_value",
    )
    use_benchmark_ensemble: bool = True
    eps: float = 1e-8


class PortfolioBacktestCalculator:
    """Portfolio-style backtest metrics calculator."""
    
    TOP_RET = "top_ret"
    TOP_RET_STD = "top_ret_std"
    TOP_RET_STABILITY = "top_ret_stability"
    MODEL_BENCHMARK_CORR = "model_benchmark_corr"
    TOP_RET_EXCESS = "top_ret_excess"
    TOP_RET_RELATIVE_IMPROVE_PCT = "top_ret_relative_improve_pct"
    BENCHMARK_TOP_RET = "benchmark_top_ret"
    BENCHMARK_TOP_RET_STD = "benchmark_top_ret_std"
    BENCHMARK_TOP_RET_STABILITY = "benchmark_top_ret_stability"

    METRIC_NAMES = {
        TOP_RET, TOP_RET_STD, TOP_RET_STABILITY, MODEL_BENCHMARK_CORR,
        TOP_RET_EXCESS, TOP_RET_RELATIVE_IMPROVE_PCT,
        BENCHMARK_TOP_RET, BENCHMARK_TOP_RET_STD, BENCHMARK_TOP_RET_STABILITY,
    }

    def __init__(self, config: Optional[PortfolioBacktestConfig] = None) -> None:
        self.config = config or PortfolioBacktestConfig()

    @staticmethod
    def _extract_frame_column(frame: Optional[pl.DataFrame], col: str) -> Optional[np.ndarray]:
        if frame is not None and col in frame.columns:
            return frame[col].to_numpy()
        return None

    def _extract_column_candidates(
        self, view: SplitView, candidates: Sequence[str],
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        for col in candidates:
            val = self._extract_frame_column(view.keys, col)
            if val is None:
                val = self._extract_frame_column(view.extra, col)
            if val is not None:
                return np.asarray(val), col
        return None, None

    @staticmethod
    def _as_float(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float64).ravel()
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float) -> Optional[float]:
        if a.size < 2 or b.size < 2:
            return None
        a_std, b_std = float(np.std(a)), float(np.std(b))
        if a_std <= eps or b_std <= eps:
            return None
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if np.isfinite(corr) else None

    def _zscore_by_day(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float64)
        for day in np.unique(dates):
            mask = dates == day
            if not np.any(mask):
                continue
            x = values[mask]
            mean, std = np.nanmean(x), np.nanstd(x)
            if not np.isfinite(mean) or not np.isfinite(std) or std <= self.config.eps:
                out[mask] = 0.0
            else:
                out[mask] = (x - mean) / (std + self.config.eps)
        return out

    def _resolve_benchmark_score(self, view: SplitView, dates: np.ndarray) -> Optional[np.ndarray]:
        if self.config.use_benchmark_ensemble:
            members = []
            for col in self.config.benchmark_member_cols:
                arr, _ = self._extract_column_candidates(view, (col,))
                if arr is not None:
                    members.append(self._as_float(arr))
            if len(members) >= 2:
                z_members = [self._zscore_by_day(m, dates) for m in members]
                return np.nanmean(np.column_stack(z_members), axis=1)

        benchmark, _ = self._extract_column_candidates(view, self.config.benchmark_col_candidates)
        return self._as_float(benchmark) if benchmark is not None else None

    def _daily_top_returns(
        self, score: np.ndarray, ret: np.ndarray, liquidity: np.ndarray, dates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        day_list, daily_returns = [], []
        for day in np.unique(dates):
            mask = dates == day
            if not np.any(mask):
                continue
            score_day, ret_day, liq_day = score[mask], ret[mask], liquidity[mask]
            order = np.argsort(-score_day, kind="mergesort")
            top_k = min(self.config.top_k, len(order))

            total_hold, total_earned = 0.0, 0.0
            for idx in order[:top_k]:
                remain = self.config.money - total_hold
                if remain < self.config.min_trade_money:
                    break
                liq = float(liq_day[idx]) if np.isfinite(liq_day[idx]) else 0.0
                if liq <= 0.0:
                    continue
                hold_money = min(remain, liq)
                total_hold += hold_money
                total_earned += float(ret_day[idx]) * self.config.ret_scale * hold_money

            day_list.append(int(day))
            daily_returns.append(total_earned / self.config.money if self.config.money > 0 else 0.0)

        return np.asarray(day_list, dtype=np.int32), np.asarray(daily_returns, dtype=np.float64)

    def _daily_model_benchmark_corr(
        self, score: np.ndarray, benchmark_score: np.ndarray, dates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        day_list, corr_list = [], []
        for day in np.unique(dates):
            mask = dates == day
            corr = self._safe_corr(score[mask], benchmark_score[mask], self.config.eps)
            if corr is not None:
                day_list.append(int(day))
                corr_list.append(corr)
        return np.asarray(day_list, dtype=np.int32), np.asarray(corr_list, dtype=np.float64)

    def compute(self, pred: np.ndarray, view: SplitView) -> Tuple[Dict[str, float], Dict[str, pl.Series]]:
        metrics: Dict[str, float] = {}
        series: Dict[str, pl.Series] = {}

        dates_raw, _ = self._extract_column_candidates(view, ("date",))
        if dates_raw is None:
            return metrics, series
        dates = np.asarray(dates_raw, dtype=np.int64).ravel()

        liquidity_raw, _ = self._extract_column_candidates(view, self.config.liquidity_col_candidates)
        if liquidity_raw is None:
            return metrics, series
        liquidity = self._as_float(liquidity_raw)

        ret_raw, _ = self._extract_column_candidates(view, self.config.ret_col_candidates)
        ret = self._as_float(ret_raw if ret_raw is not None else np.asarray(view.y).ravel())

        model_score = self._as_float(np.asarray(pred).ravel())
        model_dates, model_daily_ret = self._daily_top_returns(model_score, ret, liquidity, dates)
        
        if model_daily_ret.size > 0:
            model_mean, model_std = float(np.mean(model_daily_ret)), float(np.std(model_daily_ret))
            metrics[self.TOP_RET] = model_mean
            metrics[self.TOP_RET_STD] = model_std
            metrics[self.TOP_RET_STABILITY] = model_mean / model_std if model_std > self.config.eps else 0.0
            series["daily_top_ret_date"] = pl.Series("daily_top_ret_date", model_dates)
            series["daily_top_ret"] = pl.Series("daily_top_ret", model_daily_ret)
            series["daily_top_ret_cum_date"] = pl.Series("daily_top_ret_cum_date", model_dates)
            series["daily_top_ret_cum"] = pl.Series("daily_top_ret_cum", np.cumsum(model_daily_ret))

        benchmark_score = self._resolve_benchmark_score(view, dates)
        if benchmark_score is None:
            return metrics, series

        corr_dates, daily_corr = self._daily_model_benchmark_corr(model_score, benchmark_score, dates)
        if daily_corr.size > 0:
            metrics[self.MODEL_BENCHMARK_CORR] = float(np.mean(daily_corr))
            series["daily_model_benchmark_corr_date"] = pl.Series("daily_model_benchmark_corr_date", corr_dates)
            series["daily_model_benchmark_corr"] = pl.Series("daily_model_benchmark_corr", daily_corr)

        bench_dates, bench_daily_ret = self._daily_top_returns(benchmark_score, ret, liquidity, dates)
        if bench_daily_ret.size == 0:
            return metrics, series

        bench_mean, bench_std = float(np.mean(bench_daily_ret)), float(np.std(bench_daily_ret))
        metrics[self.BENCHMARK_TOP_RET] = bench_mean
        metrics[self.BENCHMARK_TOP_RET_STD] = bench_std
        metrics[self.BENCHMARK_TOP_RET_STABILITY] = bench_mean / bench_std if bench_std > self.config.eps else 0.0
        series["daily_benchmark_top_ret_date"] = pl.Series("daily_benchmark_top_ret_date", bench_dates)
        series["daily_benchmark_top_ret"] = pl.Series("daily_benchmark_top_ret", bench_daily_ret)
        series["daily_benchmark_top_ret_cum_date"] = pl.Series("daily_benchmark_top_ret_cum_date", bench_dates)
        series["daily_benchmark_top_ret_cum"] = pl.Series("daily_benchmark_top_ret_cum", np.cumsum(bench_daily_ret))

        model_mean_val = metrics.get(self.TOP_RET)
        if model_mean_val is not None:
            metrics[self.TOP_RET_EXCESS] = float(model_mean_val - bench_mean)
            denom = max(abs(bench_mean), self.config.relative_improve_denom_floor)
            metrics[self.TOP_RET_RELATIVE_IMPROVE_PCT] = (model_mean_val - bench_mean) / denom * self.config.relative_improve_output_scale

        common_dates, model_idx, bench_idx = np.intersect1d(model_dates, bench_dates, return_indices=True)
        if common_dates.size > 0:
            excess_daily = model_daily_ret[model_idx] - bench_daily_ret[bench_idx]
            series["daily_top_ret_excess_date"] = pl.Series("daily_top_ret_excess_date", common_dates)
            series["daily_top_ret_excess"] = pl.Series("daily_top_ret_excess", excess_daily)
            series["daily_top_ret_excess_cum_date"] = pl.Series("daily_top_ret_excess_cum_date", common_dates)
            series["daily_top_ret_excess_cum"] = pl.Series("daily_top_ret_excess_cum", np.cumsum(excess_daily))

        return metrics, series


# =============================================================================
# Regression Evaluator
# =============================================================================

class RegressionEvaluator(BaseEvaluator):
    """Generic regression evaluator for models exposing a predict() method."""
    
    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        portfolio_top_k: int = 500,
        portfolio_money: float = 1.5e9,
        portfolio_min_trade_money: float = 1.0,
        portfolio_ret_scale: float = 1.0,
        portfolio_relative_improve_denom_floor: float = 0.01,
        portfolio_relative_improve_output_scale: float = 1.0,
        ret_col_candidates: Optional[List[str]] = None,
        liquidity_col_candidates: Optional[List[str]] = None,
        benchmark_col_candidates: Optional[List[str]] = None,
        benchmark_member_cols: Optional[List[str]] = None,
        use_benchmark_ensemble: bool = True,
    ):
        self.metric_names = metric_names or [
            "pearsonr_ic", "pearsonr_icir", "top_ret", "top_ret_std",
            "top_ret_stability", "model_benchmark_corr",
            "top_ret_excess", "top_ret_relative_improve_pct",
        ]
        self.portfolio_backtest = PortfolioBacktestCalculator(
            PortfolioBacktestConfig(
                top_k=portfolio_top_k,
                money=portfolio_money,
                min_trade_money=portfolio_min_trade_money,
                ret_scale=portfolio_ret_scale,
                relative_improve_denom_floor=portfolio_relative_improve_denom_floor,
                relative_improve_output_scale=portfolio_relative_improve_output_scale,
                ret_col_candidates=tuple(ret_col_candidates or ("ret__ret_value", "ret_value", "ret")),
                liquidity_col_candidates=tuple(liquidity_col_candidates or ("liquidity__liquidity_value", "liquidity_value", "liquidity")),
                benchmark_col_candidates=tuple(benchmark_col_candidates or ("score__score_value", "benchmark__benchmark_value", "score_value", "benchmark_value")),
                benchmark_member_cols=tuple(benchmark_member_cols or ("bench1__bench1_value", "bench2__bench2_value", "bench3__bench3_value", "bench4__bench4_value", "bench5__bench5_value", "bench6__bench6_value")),
                use_benchmark_ensemble=use_benchmark_ensemble,
            )
        )

    def evaluate(
        self, model: Any, views: "ProcessedViews", modes: Optional[List[str]] = None,
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
                backtest_metrics, backtest_series = self.portfolio_backtest.compute(predictions, view)

            metrics: Dict[str, float] = {}
            for metric_name in self.metric_names:
                if metric_name in registry_names:
                    metric = MetricRegistry.get(metric_name)
                    metrics[metric_name] = metric.compute(predictions, view)
                elif metric_name in backtest_names:
                    metrics[metric_name] = float(backtest_metrics.get(metric_name, np.nan))
                else:
                    raise ValueError(f"Metric '{metric_name}' is not supported by RegressionEvaluator.")

            series = self._compute_series(predictions, view)
            series.update(backtest_series)
            results[mode] = EvalResult(metrics=metrics, series=series, predictions=predictions, mode=mode)

        return results

    @staticmethod
    def _predict(model: Any, X: np.ndarray) -> np.ndarray:
        model_cls = model.__class__
        if model_cls.__module__ == "xgboost.core" and model_cls.__name__ == "Booster":
            best_iteration = int(getattr(model, "best_iteration", -1) or -1)
            iteration_range = (0, best_iteration + 1) if best_iteration >= 0 else (0, 0)
            return np.asarray(model.inplace_predict(X, iteration_range=iteration_range)).ravel()
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X)).ravel()
        if callable(model):
            return np.asarray(model(X)).ravel()
        raise ValueError("Model does not implement predict().")

    def _compute_series(self, pred: np.ndarray, view: SplitView) -> Dict[str, pl.Series]:
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
            pred_day, true_day = pred[mask], y_true[mask]
            if np.std(pred_day) < 1e-8 or np.std(true_day) < 1e-8:
                continue
            ic = float(np.corrcoef(pred_day, true_day)[0, 1])
            if np.isfinite(ic):
                daily_ic_dates.append(int(day))
                daily_ic_values.append(ic)

        return {
            "daily_ic": pl.Series("daily_ic", daily_ic_values),
            "daily_ic_date": pl.Series("daily_ic_date", daily_ic_dates),
        }
