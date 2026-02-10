"""
指标系统 - 训练内外统一口径
"""



from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from scipy import stats

from ..data.data_dataclasses import DataBundle


def _to_numpy(x):
    """转换为 numpy 数组，兼容 numpy/pandas/polars"""
    if hasattr(x, 'to_numpy'):  # polars Series
        return x.to_numpy()
    if hasattr(x, 'values'):  # pandas Series
        return x.values
    return np.asarray(x)


class Metric(ABC):
    """指标基类 - 统一定义，避免口径漂移"""
    name: str = "base_metric"
    higher_is_better: bool = True
    
    @abstractmethod
    def compute(self, pred: np.ndarray, bundle: DataBundle) -> float:
        """计算指标"""
        pass


class PearsonICMetric(Metric):
    """Pearson IC指标"""
    name = "pearsonr_ic"
    higher_is_better = True
    
    def compute(self, pred: np.ndarray, bundle: DataBundle) -> float:
        pred = np.asarray(pred).ravel()
        y_true = bundle.y.ravel()
        date = _to_numpy(bundle.extra["date"])
        
        unique_dates = np.unique(date)
        daily_ics = []
        
        for d in unique_dates:
            mask = date == d
            if mask.sum() < 2:
                continue
            pred_d, true_d = pred[mask], y_true[mask]
            if np.std(pred_d) < 1e-8 or np.std(true_d) < 1e-8:
                continue
            ic, _ = stats.pearsonr(pred_d, true_d)
            if np.isfinite(ic):
                daily_ics.append(ic)
        
        return float(np.mean(daily_ics)) if daily_ics else 0.0


class ICIRMetric(Metric):
    """ICIR指标"""
    name = "pearsonr_icir"
    higher_is_better = True
    
    def compute(self, pred: np.ndarray, bundle: DataBundle) -> float:
        pred = np.asarray(pred).ravel()
        y_true = bundle.y.ravel()
        date = _to_numpy(bundle.extra["date"])
        
        unique_dates = np.unique(date)
        daily_ics = []
        
        for d in unique_dates:
            mask = date == d
            if mask.sum() < 2:
                continue
            pred_d, true_d = pred[mask], y_true[mask]
            if np.std(pred_d) < 1e-8 or np.std(true_d) < 1e-8:
                continue
            ic, _ = stats.pearsonr(pred_d, true_d)
            if np.isfinite(ic):
                daily_ics.append(ic)
        
        if not daily_ics:
            return 0.0
        ic_mean = np.mean(daily_ics)
        ic_std = np.std(daily_ics)
        return float(ic_mean / ic_std) if ic_std > 1e-8 else 0.0


class MSEMetric(Metric):
    """MSE指标"""
    name = "mse"
    higher_is_better = False
    
    def compute(self, pred: np.ndarray, bundle: DataBundle) -> float:
        pred = np.asarray(pred).ravel()
        y_true = bundle.y.ravel()
        return float(np.mean((pred - y_true) ** 2))


class PortfolioReturnMetric(Metric):
    """组合收益指标"""
    name = "portfolio_ret"
    higher_is_better = True
    
    def __init__(self, max_positions: int = 500, money: float = 1.5e9,
                 ret_col: str = "ret", liquidity_col: str = "liquidity"):
        self.max_positions = max_positions
        self.money = money
        self.ret_col = ret_col
        self.liquidity_col = liquidity_col
    
    def compute(self, pred: np.ndarray, bundle: DataBundle) -> float:
        pred = np.asarray(pred).ravel()
        date = _to_numpy(bundle.extra["date"])
        ret = _to_numpy(bundle.extra[self.ret_col])
        liquidity = _to_numpy(bundle.extra[self.liquidity_col])
        
        unique_dates = np.unique(date)
        daily_rets = []
        
        for d in unique_dates:
            mask = date == d
            if not np.any(mask):
                continue
            
            pred_d = pred[mask]
            ret_d = ret[mask]
            liq_d = liquidity[mask]
            
            order = np.argsort(-pred_d)
            ret_sorted = ret_d[order]
            liq_sorted = liq_d[order]
            
            total_hold = 0.0
            total_earned = 0.0
            k = min(self.max_positions, len(order))
            
            for i in range(k):
                if (self.money - total_hold) < 1e-6:
                    break
                liq_i = float(liq_sorted[i]) if np.isfinite(liq_sorted[i]) else 0.0
                if liq_i <= 0:
                    continue
                hold_money = min(self.money - total_hold, liq_i)
                total_hold += hold_money
                total_earned += float(ret_sorted[i]) * hold_money
            
            daily_rets.append(total_earned / self.money if self.money > 0 else 0.0)
        
        return float(np.mean(daily_rets)) if daily_rets else 0.0


class MetricRegistry:
    """指标注册表"""
    _registry: Dict[str, Metric] = {}
    
    @classmethod
    def register(cls, metric: Metric) -> None:
        cls._registry[metric.name] = metric
    
    @classmethod
    def get(cls, name: str) -> Metric:
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._registry.keys())


# 注册默认指标
MetricRegistry.register(PearsonICMetric())
MetricRegistry.register(ICIRMetric())
MetricRegistry.register(MSEMetric())
