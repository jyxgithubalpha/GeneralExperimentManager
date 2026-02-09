"""
LightGBMEvaluator - LightGBM 模型评估器
"""


from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews

from ..base import BaseEvaluator
from ..training_dataclasses import EvalResult


class LightGBMEvaluator(BaseEvaluator):
    """
    LightGBM 模型评估器
    
    计算各数据集上的指标和时间序列
    """
    
    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
    ):
        self.metric_names = metric_names or ["pearsonr_ic", "pearsonr_icir"]
    
    def evaluate(
        self,
        model: Any,
        views: "ProcessedViews",
        modes: Optional[List[str]] = None,
    ) -> Dict[str, EvalResult]:
        """
        评估 LightGBM 模型
        
        Args:
            model: 训练好的 LightGBM 模型
            views: 处理后的视图
            modes: 要评估的数据集 ["train", "val", "test"]
            
        Returns:
            {mode: EvalResult} 字典
        """
        from ...utils.metrics import MetricRegistry
        
        modes = modes or ["train", "val", "test"]
        results = {}
        
        for mode in modes:
            view = views.get(mode)
            pred = model.predict(view.X)
            
            # 计算指标
            metrics = {}
            for name in self.metric_names:
                metric = MetricRegistry.get(name)
                metrics[f"{mode}_{name}"] = metric.compute(pred, view)
            
            # 计算时间序列 IC
            series = self._compute_series(pred, view)
            
            results[mode] = EvalResult(
                metrics=metrics,
                series=series,
                predictions=pred,
            )
        
        return results
    
    def _compute_series(self, pred: np.ndarray, view) -> Dict[str, pl.Series]:
        """计算时间序列指标"""
        from scipy import stats
        
        pred = np.asarray(pred).ravel()
        y_true = view.y.ravel()
        dates = view.keys["date"].to_numpy()
        
        daily_ics = []
        daily_dates = []
        
        for d in np.unique(dates):
            mask = dates == d
            if mask.sum() < 2:
                continue
            pred_d, true_d = pred[mask], y_true[mask]
            if np.std(pred_d) < 1e-8 or np.std(true_d) < 1e-8:
                continue
            ic, _ = stats.pearsonr(pred_d, true_d)
            if np.isfinite(ic):
                daily_ics.append(ic)
                daily_dates.append(d)
        
        return {
            "daily_ic": pl.Series("daily_ic", daily_ics),
        }
