"""
Metric registry used by both LightGBM feval and offline evaluator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from scipy import stats


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "values"):
        return x.values
    return np.asarray(x)


def _extract_column(bundle: Any, column: str) -> np.ndarray:
    # Priority: keys -> extra -> meta
    if hasattr(bundle, "keys") and bundle.keys is not None and column in bundle.keys.columns:
        return _to_numpy(bundle.keys[column])
    if hasattr(bundle, "extra") and bundle.extra is not None and column in bundle.extra.columns:
        return _to_numpy(bundle.extra[column])
    if hasattr(bundle, "meta") and bundle.meta is not None and column in bundle.meta.columns:
        return _to_numpy(bundle.meta[column])
    raise ValueError(f"Column '{column}' was not found in keys/extra/meta.")


def _extract_y(bundle: Any) -> np.ndarray:
    if not hasattr(bundle, "y"):
        raise ValueError("Bundle has no 'y' attribute.")
    return _to_numpy(bundle.y).ravel()


def _daily_pearson_ic(pred: np.ndarray, y_true: np.ndarray, dates: np.ndarray) -> np.ndarray:
    daily_ics = []
    for day in np.unique(dates):
        mask = dates == day
        if mask.sum() < 2:
            continue

        pred_day = pred[mask]
        y_day = y_true[mask]
        if np.std(pred_day) < 1e-8 or np.std(y_day) < 1e-8:
            continue

        ic, _ = stats.pearsonr(pred_day, y_day)
        if np.isfinite(ic):
            daily_ics.append(ic)

    return np.asarray(daily_ics, dtype=np.float64)


class Metric(ABC):
    name: str = "base_metric"
    higher_is_better: bool = True

    @abstractmethod
    def compute(self, pred: np.ndarray, bundle: Any) -> float:
        pass


class PearsonICMetric(Metric):
    name = "pearsonr_ic"
    higher_is_better = True

    def compute(self, pred: np.ndarray, bundle: Any) -> float:
        pred = np.asarray(pred).ravel()
        y_true = _extract_y(bundle)
        dates = _extract_column(bundle, "date")

        daily_ics = _daily_pearson_ic(pred, y_true, dates)
        if daily_ics.size == 0:
            return 0.0
        return float(np.mean(daily_ics))


class ICIRMetric(Metric):
    name = "pearsonr_icir"
    higher_is_better = True

    def compute(self, pred: np.ndarray, bundle: Any) -> float:
        pred = np.asarray(pred).ravel()
        y_true = _extract_y(bundle)
        dates = _extract_column(bundle, "date")

        daily_ics = _daily_pearson_ic(pred, y_true, dates)
        if daily_ics.size == 0:
            return 0.0

        ic_mean = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics))
        return ic_mean / ic_std if ic_std > 1e-8 else 0.0


class MSEMetric(Metric):
    name = "mse"
    higher_is_better = False

    def compute(self, pred: np.ndarray, bundle: Any) -> float:
        pred = np.asarray(pred).ravel()
        y_true = _extract_y(bundle)
        return float(np.mean((pred - y_true) ** 2))


class PortfolioReturnMetric(Metric):
    name = "portfolio_ret"
    higher_is_better = True

    def __init__(
        self,
        max_positions: int = 500,
        money: float = 1.5e9,
        ret_col: str = "ret",
        liquidity_col: str = "liquidity",
    ):
        self.max_positions = max_positions
        self.money = money
        self.ret_col = ret_col
        self.liquidity_col = liquidity_col

    def compute(self, pred: np.ndarray, bundle: Any) -> float:
        pred = np.asarray(pred).ravel()
        dates = _extract_column(bundle, "date")
        ret = _extract_column(bundle, self.ret_col)
        liquidity = _extract_column(bundle, self.liquidity_col)

        daily_rets = []
        for day in np.unique(dates):
            mask = dates == day
            if not np.any(mask):
                continue

            pred_day = pred[mask]
            ret_day = ret[mask]
            liq_day = liquidity[mask]

            order = np.argsort(-pred_day)
            ret_sorted = ret_day[order]
            liq_sorted = liq_day[order]

            total_hold = 0.0
            total_earned = 0.0

            top_k = min(self.max_positions, len(order))
            for idx in range(top_k):
                if self.money - total_hold < 1e-6:
                    break

                liq = float(liq_sorted[idx]) if np.isfinite(liq_sorted[idx]) else 0.0
                if liq <= 0:
                    continue

                hold_money = min(self.money - total_hold, liq)
                total_hold += hold_money
                total_earned += float(ret_sorted[idx]) * hold_money

            daily_rets.append(total_earned / self.money if self.money > 0 else 0.0)

        if not daily_rets:
            return 0.0
        return float(np.mean(daily_rets))


class MetricRegistry:
    _registry: Dict[str, Metric] = {}

    @classmethod
    def register(cls, metric: Metric) -> None:
        cls._registry[metric.name] = metric

    @classmethod
    def get(cls, name: str) -> Metric:
        if name not in cls._registry:
            raise ValueError(
                f"Metric '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._registry.keys())


MetricRegistry.register(PearsonICMetric())
MetricRegistry.register(ICIRMetric())
MetricRegistry.register(MSEMetric())
