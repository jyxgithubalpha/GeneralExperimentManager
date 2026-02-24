"""
Portfolio-style backtest metrics for evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from ...data.data_dataclasses import SplitView


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    top_k: int = 500
    money: float = 1.5e9
    min_trade_money: float = 1.0
    ret_scale: float = 100.0
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
    TOP_RET = "top_ret"
    TOP_RET_STD = "top_ret_std"
    MODEL_BENCHMARK_CORR = "model_benchmark_corr"
    TOP_RET_RELATIVE_IMPROVE_PCT = "top_ret_relative_improve_pct"
    BENCHMARK_TOP_RET = "benchmark_top_ret"
    BENCHMARK_TOP_RET_STD = "benchmark_top_ret_std"

    METRIC_NAMES = {
        TOP_RET,
        TOP_RET_STD,
        MODEL_BENCHMARK_CORR,
        TOP_RET_RELATIVE_IMPROVE_PCT,
        BENCHMARK_TOP_RET,
        BENCHMARK_TOP_RET_STD,
    }

    def __init__(self, config: Optional[PortfolioBacktestConfig] = None) -> None:
        self.config = config or PortfolioBacktestConfig()

    @staticmethod
    def _extract_frame_column(frame: Optional[pl.DataFrame], col: str) -> Optional[np.ndarray]:
        if frame is not None and col in frame.columns:
            return frame[col].to_numpy()
        return None

    def _extract_column_candidates(
        self,
        view: SplitView,
        candidates: Sequence[str],
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
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std <= eps or b_std <= eps:
            return None
        corr = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(corr):
            return None
        return float(corr)

    def _zscore_by_day(self, values: np.ndarray, dates: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float64)
        for day in np.unique(dates):
            mask = dates == day
            if not np.any(mask):
                continue
            x = values[mask]
            mean = np.nanmean(x)
            std = np.nanstd(x)
            if not np.isfinite(mean) or not np.isfinite(std) or std <= self.config.eps:
                out[mask] = 0.0
            else:
                out[mask] = (x - mean) / (std + self.config.eps)
        return out

    def _resolve_benchmark_score(
        self,
        view: SplitView,
        dates: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.config.use_benchmark_ensemble:
            members = []
            for col in self.config.benchmark_member_cols:
                arr, _ = self._extract_column_candidates(view, (col,))
                if arr is not None:
                    members.append(self._as_float(arr))
            if len(members) >= 2:
                z_members = [self._zscore_by_day(member, dates) for member in members]
                stacked = np.column_stack(z_members)
                return np.nanmean(stacked, axis=1)

        benchmark, _ = self._extract_column_candidates(view, self.config.benchmark_col_candidates)
        if benchmark is None:
            return None
        return self._as_float(benchmark)

    def _daily_top_returns(
        self,
        score: np.ndarray,
        ret: np.ndarray,
        liquidity: np.ndarray,
        dates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        day_list = []
        daily_returns = []

        for day in np.unique(dates):
            mask = dates == day
            if not np.any(mask):
                continue

            score_day = score[mask]
            ret_day = ret[mask]
            liq_day = liquidity[mask]

            order = np.argsort(-score_day, kind="mergesort")
            top_k = min(self.config.top_k, len(order))

            total_hold = 0.0
            total_earned = 0.0
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
        self,
        score: np.ndarray,
        benchmark_score: np.ndarray,
        dates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        day_list = []
        corr_list = []

        for day in np.unique(dates):
            mask = dates == day
            corr = self._safe_corr(score[mask], benchmark_score[mask], self.config.eps)
            if corr is None:
                continue
            day_list.append(int(day))
            corr_list.append(corr)

        return np.asarray(day_list, dtype=np.int32), np.asarray(corr_list, dtype=np.float64)

    def compute(
        self,
        pred: np.ndarray,
        view: SplitView,
    ) -> Tuple[Dict[str, float], Dict[str, pl.Series]]:
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
        if ret_raw is None:
            ret_raw = np.asarray(view.y).ravel()
        ret = self._as_float(ret_raw)

        model_score = self._as_float(np.asarray(pred).ravel())
        model_dates, model_daily_ret = self._daily_top_returns(model_score, ret, liquidity, dates)
        if model_daily_ret.size > 0:
            metrics[self.TOP_RET] = float(np.mean(model_daily_ret))
            metrics[self.TOP_RET_STD] = float(np.std(model_daily_ret))
            series["daily_top_ret_date"] = pl.Series("daily_top_ret_date", model_dates)
            series["daily_top_ret"] = pl.Series("daily_top_ret", model_daily_ret)

        benchmark_score = self._resolve_benchmark_score(view, dates)
        if benchmark_score is None:
            return metrics, series

        corr_dates, daily_corr = self._daily_model_benchmark_corr(model_score, benchmark_score, dates)
        if daily_corr.size > 0:
            metrics[self.MODEL_BENCHMARK_CORR] = float(np.mean(daily_corr))
            series["daily_model_benchmark_corr_date"] = pl.Series(
                "daily_model_benchmark_corr_date",
                corr_dates,
            )
            series["daily_model_benchmark_corr"] = pl.Series(
                "daily_model_benchmark_corr",
                daily_corr,
            )

        bench_dates, bench_daily_ret = self._daily_top_returns(benchmark_score, ret, liquidity, dates)
        if bench_daily_ret.size == 0:
            return metrics, series

        bench_mean = float(np.mean(bench_daily_ret))
        bench_std = float(np.std(bench_daily_ret))
        metrics[self.BENCHMARK_TOP_RET] = bench_mean
        metrics[self.BENCHMARK_TOP_RET_STD] = bench_std
        series["daily_benchmark_top_ret_date"] = pl.Series("daily_benchmark_top_ret_date", bench_dates)
        series["daily_benchmark_top_ret"] = pl.Series("daily_benchmark_top_ret", bench_daily_ret)

        model_mean = metrics.get(self.TOP_RET)
        if model_mean is not None:
            denom = abs(bench_mean)
            if denom > self.config.eps:
                rel = (model_mean - bench_mean) / denom * 100.0
            else:
                rel = 0.0
            metrics[self.TOP_RET_RELATIVE_IMPROVE_PCT] = float(rel)

        return metrics, series
