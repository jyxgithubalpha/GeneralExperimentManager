"""
DataProcessor - Split-aware 数据处理器

负责:
- 接收 SplitViews 与 RollingState
- 对 search/train 做 fit，对 val/eval 做 transform
- 包含: feature weighting, 动态因子筛选, zscore/winsorize

核心原则:
- 必须在 split 内执行，避免数据泄漏
- fit 只能用训练集数据
- 产生 feature_mask 和 transform_state
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..experiment.state_dataclasses import (
    DataWeightState,
    RollingState,
)
from ..data.data_dataclasses import SplitView, SplitViews, ProcessedViews, TransformState



# =============================================================================
# 1. Transform 基类
# =============================================================================

class Transform(ABC):
    """
    变换基类 - fit/transform 分离防泄露
    
    Attributes:
        name: 变换名称
        target: 变换目标 ("X", "y", "both")
    """
    name: str = "base_transform"
    
    def __init__(self, target: str = "X"):
        """
        Args:
            target: 变换目标，"X" 处理特征，"y" 处理标签，"both" 同时处理
        """
        if target not in ("X", "y", "both"):
            raise ValueError(f"target must be 'X', 'y', or 'both', got {target}")
        self.target = target
    
    @abstractmethod
    def fit(self, view: SplitView) -> "Transform":
        """只能用 train 数据 fit"""
        return self
    
    @abstractmethod
    def transform(self, view: SplitView) -> SplitView:
        """变换数据"""
        pass
    
    def fit_transform(self, view: SplitView) -> SplitView:
        return self.fit(view).transform(view)
    
    def get_state(self) -> Dict[str, Any]:
        """获取可序列化状态"""
        return {}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """设置状态"""
        pass
    
    def _get_data(self, view: SplitView) -> np.ndarray:
        """根据 target 获取要处理的数据"""
        if self.target == "X":
            return view.X
        elif self.target == "y":
            return view.y
        else:  # both
            return np.hstack([view.X, view.y])
    
    def _set_data(self, view: SplitView, data: np.ndarray) -> SplitView:
        """根据 target 设置处理后的数据"""
        if self.target == "X":
            return SplitView(
                indices=view.indices,
                X=data,
                y=view.y,
                keys=view.keys,
                feature_names=view.feature_names,
                label_names=view.label_names,
                extra=view.extra,
                group=view.group,
            )
        elif self.target == "y":
            return SplitView(
                indices=view.indices,
                X=view.X,
                y=data,
                keys=view.keys,
                feature_names=view.feature_names,
                label_names=view.label_names,
                extra=view.extra,
                group=view.group,
            )
        else:  # both
            n_features = view.X.shape[1]
            return SplitView(
                indices=view.indices,
                X=data[:, :n_features],
                y=data[:, n_features:],
                keys=view.keys,
                feature_names=view.feature_names,
                label_names=view.label_names,
                extra=view.extra,
                group=view.group,
            )


class IdentityTransform(Transform):
    """恒等变换"""
    name = "identity"
    
    def __init__(self):
        super().__init__(target="X")
    
    def fit(self, view: SplitView) -> "IdentityTransform":
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        return view


# =============================================================================
# 2. 标准化/归一化变换
# =============================================================================

class StandardizeTransform(Transform):
    """
    标准化变换 (Z-Score)
    
    fit: 计算 train 的 mean/std
    transform: 使用 train 的统计量标准化
    
    Args:
        target: 变换目标 ("X", "y", "both")
        eps: 防止除零的小数
        per_date: 是否按日期分组计算统计量
    """
    name = "standardize"
    
    def __init__(self, target: str = "X", eps: float = 1e-8, per_date: bool = True):
        super().__init__(target=target)
        self.eps = eps
        self.per_date = per_date
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._date_stats: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
    
    def fit(self, view: SplitView) -> "StandardizeTransform":
        data = self._get_data(view)
        
        if self.per_date:
            # 按日期分组计算统计量
            self._date_stats = {}
            dates = view.keys["date"].values
            for d in np.unique(dates):
                mask = dates == d
                data_d = data[mask]
                self._date_stats[d] = (np.mean(data_d, axis=0), np.std(data_d, axis=0))
            # 计算全局统计量作为 fallback
            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0)
        else:
            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0)
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        data = self._get_data(view).copy()
        
        if self.per_date and self._date_stats is not None:
            dates = view.keys["date"].values
            for d in np.unique(dates):
                mask = dates == d
                if d in self._date_stats:
                    mean, std = self._date_stats[d]
                else:
                    mean, std = self._mean, self._std
                data[mask] = (data[mask] - mean) / (std + self.eps)
        else:
            data = (data - self._mean) / (self._std + self.eps)
        
        return self._set_data(view, data)
    
    def get_state(self) -> Dict[str, Any]:
        return {"mean": self._mean, "std": self._std, "date_stats": self._date_stats}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self._mean = state.get("mean")
        self._std = state.get("std")
        self._date_stats = state.get("date_stats")


class RankTransform(Transform):
    """
    排名变换 - 截面排名转百分比
    
    Args:
        target: 变换目标 ("X", "y", "both")
        per_date: 是否按日期分组计算排名
    """
    name = "rank"
    
    def __init__(self, target: str = "X", per_date: bool = True):
        super().__init__(target=target)
        self.per_date = per_date
    
    def fit(self, view: SplitView) -> "RankTransform":
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        data = self._get_data(view).copy()
        
        if self.per_date:
            dates = view.keys["date"].values
            for d in np.unique(dates):
                mask = dates == d
                data_d = data[mask]
                for j in range(data_d.shape[1]):
                    col = data_d[:, j]
                    ranks = np.argsort(np.argsort(col))
                    data_d[:, j] = ranks / (len(col) - 1 + 1e-8)
                data[mask] = data_d
        else:
            for j in range(data.shape[1]):
                col = data[:, j]
                ranks = np.argsort(np.argsort(col))
                data[:, j] = ranks / (len(col) - 1 + 1e-8)
        
        return self._set_data(view, data)


# =============================================================================
# 3. Winsorize 变换
# =============================================================================

class WinsorizeTransform(Transform):
    """
    Winsorize 变换 - 截尾处理
    
    Args:
        target: 变换目标 ("X", "y", "both")
        lower: 下分位数
        upper: 上分位数
        per_date: 是否按日期分组计算
    """
    name = "winsorize"
    
    def __init__(self, target: str = "X", lower: float = 0.01, upper: float = 0.99, per_date: bool = True):
        super().__init__(target=target)
        self.lower = lower
        self.upper = upper
        self.per_date = per_date
        self._lower_bounds: Optional[np.ndarray] = None
        self._upper_bounds: Optional[np.ndarray] = None
        self._date_bounds: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
    
    def fit(self, view: SplitView) -> "WinsorizeTransform":
        data = self._get_data(view)
        
        if self.per_date:
            self._date_bounds = {}
            dates = view.keys["date"].values
            for d in np.unique(dates):
                mask = dates == d
                data_d = data[mask]
                lower_b = np.percentile(data_d, self.lower * 100, axis=0)
                upper_b = np.percentile(data_d, self.upper * 100, axis=0)
                self._date_bounds[d] = (lower_b, upper_b)
            # 全局 bounds 作为 fallback
            self._lower_bounds = np.percentile(data, self.lower * 100, axis=0)
            self._upper_bounds = np.percentile(data, self.upper * 100, axis=0)
        else:
            self._lower_bounds = np.percentile(data, self.lower * 100, axis=0)
            self._upper_bounds = np.percentile(data, self.upper * 100, axis=0)
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        data = self._get_data(view).copy()
        
        if self.per_date and self._date_bounds is not None:
            dates = view.keys["date"].values
            for d in np.unique(dates):
                mask = dates == d
                if d in self._date_bounds:
                    lower_b, upper_b = self._date_bounds[d]
                else:
                    lower_b, upper_b = self._lower_bounds, self._upper_bounds
                data[mask] = np.clip(data[mask], lower_b, upper_b)
        else:
            data = np.clip(data, self._lower_bounds, self._upper_bounds)
        
        return self._set_data(view, data)
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "lower_bounds": self._lower_bounds,
            "upper_bounds": self._upper_bounds,
            "date_bounds": self._date_bounds,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self._lower_bounds = state.get("lower_bounds")
        self._upper_bounds = state.get("upper_bounds")
        self._date_bounds = state.get("date_bounds")


# =============================================================================
# 4. 特征选择变换
# =============================================================================

class FeatureSelectionTransform(Transform):
    """
    特征选择变换
    
    支持方法:
    - variance: 方差过滤
    - correlation: 与标签相关性过滤
    - topk: 保留 top-k 重要特征
    - keep_features: 保留指定特征
    """
    name = "feature_selection"
    
    def __init__(
        self,
        method: str = "variance",
        threshold: float = 0.0,
        topk: Optional[int] = None,
        keep_features: Optional[List[str]] = None,
    ):
        self.method = method
        self.threshold = threshold
        self.topk = topk
        self.keep_features = keep_features
        self._mask: Optional[np.ndarray] = None
        self._selected_names: Optional[List[str]] = None
    
    def fit(self, view: SplitView) -> "FeatureSelectionTransform":
        n_features = view.X.shape[1]
        
        if self.keep_features is not None:
            # 根据指定特征名筛选
            self._mask = np.array([n in self.keep_features for n in view.feature_names])
        elif self.method == "variance":
            variances = np.var(view.X, axis=0)
            self._mask = variances > self.threshold
        elif self.method == "correlation":
            # 计算与标签的相关性
            y = view.y.ravel()
            corrs = np.abs([np.corrcoef(view.X[:, j], y)[0, 1] for j in range(n_features)])
            corrs = np.nan_to_num(corrs, 0)
            if self.topk is not None:
                indices = np.argsort(corrs)[-self.topk:]
                self._mask = np.zeros(n_features, dtype=bool)
                self._mask[indices] = True
            else:
                self._mask = corrs > self.threshold
        elif self.method == "topk" and self.topk is not None:
            # 需要外部提供 importance，这里用方差作为默认
            variances = np.var(view.X, axis=0)
            indices = np.argsort(variances)[-self.topk:]
            self._mask = np.zeros(n_features, dtype=bool)
            self._mask[indices] = True
        else:
            self._mask = np.ones(n_features, dtype=bool)
        
        self._selected_names = [n for n, m in zip(view.feature_names, self._mask) if m]
        return self
    
    def set_importance_mask(self, importance: np.ndarray, topk: int) -> None:
        """根据外部提供的 importance 设置 mask"""
        indices = np.argsort(importance)[-topk:]
        self._mask = np.zeros(len(importance), dtype=bool)
        self._mask[indices] = True
    
    def transform(self, view: SplitView) -> SplitView:
        if self._mask is None:
            return view
        
        return SplitView(
            indices=view.indices,
            X=view.X[:, self._mask],
            y=view.y,
            keys=view.keys,
            feature_names=self._selected_names,
            label_names=view.label_names,
            extra=view.extra,
            group=view.group,
        )
    
    def get_state(self) -> Dict[str, Any]:
        return {"mask": self._mask, "selected_names": self._selected_names}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self._mask = state.get("mask")
        self._selected_names = state.get("selected_names")


# =============================================================================
# 5. 特征加权变换
# =============================================================================

class FeatureWeightingTransform(Transform):
    """
    特征加权变换
    
    使用外部提供的权重对特征进行加权
    """
    name = "feature_weighting"
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights
        self._weights: Optional[np.ndarray] = None
    
    def fit(self, view: SplitView) -> "FeatureWeightingTransform":
        if self.weights is not None:
            self._weights = self.weights
        else:
            self._weights = np.ones(view.X.shape[1])
        return self
    
    def set_weights(self, weights: np.ndarray) -> None:
        """设置权重"""
        self._weights = weights
    
    def transform(self, view: SplitView) -> SplitView:
        if self._weights is None:
            return view
        
        return SplitView(
            indices=view.indices,
            X=view.X * self._weights,
            y=view.y,
            keys=view.keys,
            feature_names=view.feature_names,
            label_names=view.label_names,
            extra=view.extra,
            group=view.group,
        )
    
    def get_state(self) -> Dict[str, Any]:
        return {"weights": self._weights}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self._weights = state.get("weights")


# =============================================================================
# 6. NaN 处理变换
# =============================================================================

class FillNaNTransform(Transform):
    """填充 NaN"""
    name = "fillna"
    
    def __init__(self, value: float = 0.0, method: str = "constant"):
        self.value = value
        self.method = method
        self._fill_values: Optional[np.ndarray] = None
    
    def fit(self, view: SplitView) -> "FillNaNTransform":
        if self.method == "mean":
            self._fill_values = np.nanmean(view.X, axis=0)
        elif self.method == "median":
            self._fill_values = np.nanmedian(view.X, axis=0)
        else:
            self._fill_values = np.full(view.X.shape[1], self.value)
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        X_new = view.X.copy()
        for j in range(X_new.shape[1]):
            mask = np.isnan(X_new[:, j])
            X_new[mask, j] = self._fill_values[j]
        
        return SplitView(
            indices=view.indices,
            X=X_new,
            y=view.y,
            keys=view.keys,
            feature_names=view.feature_names,
            label_names=view.label_names,
            extra=view.extra,
            group=view.group,
        )
    
    def get_state(self) -> Dict[str, Any]:
        return {"fill_values": self._fill_values}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self._fill_values = state.get("fill_values")


class DropNaNTransform(Transform):
    """删除包含 NaN 的样本"""
    name = "dropna"
    
    def __init__(self, how: str = "any"):
        self.how = how
    
    def fit(self, view: SplitView) -> "DropNaNTransform":
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        if self.how == "any":
            mask = ~np.any(np.isnan(view.X), axis=1)
        else:
            mask = ~np.all(np.isnan(view.X), axis=1)
        
        return SplitView(
            indices=view.indices[mask],
            X=view.X[mask],
            y=view.y[mask],
            keys=view.keys.iloc[mask].reset_index(drop=True),
            feature_names=view.feature_names,
            label_names=view.label_names,
            extra=view.extra.iloc[mask].reset_index(drop=True) if view.extra is not None else None,
            group=view.group.iloc[mask].reset_index(drop=True) if view.group is not None else None,
        )


# =============================================================================
# 7. Sample Weighting Transforms - 样本加权
# =============================================================================

class SampleWeightingTransform(Transform):
    """
    样本加权变换
    
    根据 DataWeightState 计算样本权重，包括:
    - feature_weights: 特征加权 (乘以 X)
    - asset_weights: 股票加权
    - industry_weights: 行业加权
    - time_weights: 时间加权
    
    样本权重存储在 view.extra["sample_weight"] 中
    """
    name = "sample_weighting"
    
    def __init__(
        self,
        use_feature_weights: bool = True,
        use_asset_weights: bool = True,
        use_industry_weights: bool = True,
        use_time_weights: bool = True,
        industry_col: str = "industry",
    ):
        self.use_feature_weights = use_feature_weights
        self.use_asset_weights = use_asset_weights
        self.use_industry_weights = use_industry_weights
        self.use_time_weights = use_time_weights
        self.industry_col = industry_col
        self._data_weight_state: Optional[DataWeightState] = None
    
    def set_weight_state(self, state: DataWeightState) -> None:
        """设置数据加权状态"""
        self._data_weight_state = state
    
    def fit(self, view: SplitView) -> "SampleWeightingTransform":
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        X_new = view.X.copy()
        
        # 计算样本权重
        sample_weight = np.ones(len(view.indices), dtype=np.float32)
        
        if self._data_weight_state is not None:
            # 使用 DataWeightState 计算样本权重
            sample_weight = self._data_weight_state.get_sample_weight(
                keys=view.keys,
                group=view.group,
                industry_col=self.industry_col,
            )
            
            # 应用特征权重 (乘以 X)
            if self.use_feature_weights and self._data_weight_state.feature_weights is not None:
                feature_weights = self._data_weight_state.feature_weights
                if len(feature_weights) == X_new.shape[1]:
                    X_new = X_new * feature_weights
        
        # 将样本权重存入 extra
        extra_new = view.extra.copy() if view.extra is not None else pd.DataFrame(index=range(len(view.indices)))
        extra_new["sample_weight"] = sample_weight
        
        return SplitView(
            indices=view.indices,
            X=X_new,
            y=view.y,
            keys=view.keys,
            feature_names=view.feature_names,
            label_names=view.label_names,
            extra=extra_new,
            group=view.group,
        )
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "data_weight_state": self._data_weight_state,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self._data_weight_state = state.get("data_weight_state")


class TimeDecayWeightingTransform(Transform):
    """
    时间衰减加权
    
    越新的样本权重越高
    """
    name = "time_decay_weighting"
    
    def __init__(self, decay_rate: float = 0.99, reference_date: Optional[int] = None):
        """
        Args:
            decay_rate: 每天的衰减率 (0.99 表示每天衰减 1%)
            reference_date: 参考日期 (None 表示使用最大日期)
        """
        self.decay_rate = decay_rate
        self.reference_date = reference_date
        self._max_date: Optional[int] = None
    
    def fit(self, view: SplitView) -> "TimeDecayWeightingTransform":
        if self.reference_date is None:
            self._max_date = view.keys["date"].max()
        else:
            self._max_date = self.reference_date
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        dates = view.keys["date"].values
        
        # 计算距离参考日期的天数
        days_diff = (self._max_date - dates) / 1  # 简化：假设日期差为天数
        
        # 计算时间衰减权重
        time_weights = np.power(self.decay_rate, days_diff).astype(np.float32)
        
        # 将权重存入 extra
        extra_new = view.extra.copy() if view.extra is not None else pd.DataFrame(index=range(len(view.indices)))
        if "sample_weight" in extra_new.columns:
            extra_new["sample_weight"] = extra_new["sample_weight"] * time_weights
        else:
            extra_new["sample_weight"] = time_weights
        
        return SplitView(
            indices=view.indices,
            X=view.X,
            y=view.y,
            keys=view.keys,
            feature_names=view.feature_names,
            label_names=view.label_names,
            extra=extra_new,
            group=view.group,
        )
    
    def get_state(self) -> Dict[str, Any]:
        return {"max_date": self._max_date}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        self._max_date = state.get("max_date")


# =============================================================================
# 7. DataProcessor - 主处理器
# =============================================================================

class DataProcessor:
    """
    Split-aware 数据处理器
    
    责任:
    - 接收 SplitViews 与 RollingState
    - 对 train 做 fit，对所有集合做 transform
    - 包含: feature weighting, 动态因子筛选, zscore/winsorize
    
    产生:
    - feature_mask
    - transform_state
    """
    
    def __init__(self, transforms: Optional[List[Transform]] = None):
        self.transforms = transforms or []
        self._fitted = False
        self._transform_state: Optional[TransformState] = None
        self._feature_mask: Optional[np.ndarray] = None
    
    def add_transform(self, transform: Transform) -> "DataProcessor":
        """添加变换"""
        self.transforms.append(transform)
        return self
    
    def fit(self, train_view: SplitView, rolling_state: Optional[RollingState] = None) -> "DataProcessor":
        """
        用训练数据拟合所有变换
        
        Args:
            train_view: 训练集视图
            rolling_state: 滚动状态 (可选，用于特征加权等)
        """
        # 如果有 RollingState，应用特征权重和样本权重状态
        if rolling_state is not None:
            # 应用特征重要性权重
            if rolling_state.importance_ema is not None:
                for t in self.transforms:
                    if isinstance(t, FeatureWeightingTransform):
                        t.set_weights(rolling_state.importance_ema)
                    elif isinstance(t, FeatureSelectionTransform) and t.method == "importance":
                        if t.topk is not None:
                            t.set_importance_mask(rolling_state.importance_ema, t.topk)
            
            # 应用样本加权状态
            if rolling_state.data_weight_state is not None:
                for t in self.transforms:
                    if isinstance(t, SampleWeightingTransform):
                        t.set_weight_state(rolling_state.data_weight_state)
        
        # 逐步 fit 和 transform
        view = train_view
        for t in self.transforms:
            t.fit(view)
            view = t.transform(view)
        
        # 收集 feature_mask
        for t in self.transforms:
            if isinstance(t, FeatureSelectionTransform) and t._mask is not None:
                self._feature_mask = t._mask
                break
        
        # 保存状态
        self._transform_state = TransformState(
            statistics={t.name: t.get_state() for t in self.transforms},
            feature_mask=self._feature_mask,
        )
        
        self._fitted = True
        return self
    
    def transform(self, view: SplitView) -> SplitView:
        """变换数据"""
        if not self._fitted:
            raise RuntimeError("DataProcessor not fitted. Call fit() first.")
        
        result = view
        for t in self.transforms:
            result = t.transform(result)
        return result
    
    def fit_transform(
        self,
        split_views: SplitViews,
        rolling_state: Optional[RollingState] = None,
    ) -> ProcessedViews:
        """
        拟合并变换所有数据集
        
        Args:
            split_views: Split 视图集合
            rolling_state: 滚动状态
            
        Returns:
            ProcessedViews 实例
        """
        # 用 train 拟合
        self.fit(split_views.train, rolling_state)
        
        # 变换所有数据集
        train_processed = self.transform(split_views.train)
        val_processed = self.transform(split_views.val)
        test_processed = self.transform(split_views.test)
        
        return ProcessedViews(
            train=train_processed,
            val=val_processed,
            test=test_processed,
            split_spec=split_views.split_spec,
            feature_mask=self._feature_mask,
            transform_state=self._transform_state,
        )
    
    def get_state(self) -> TransformState:
        """获取变换状态"""
        if self._transform_state is None:
            return TransformState()
        return self._transform_state
    
    def set_state(self, state: TransformState) -> None:
        """设置变换状态"""
        self._transform_state = state
        self._feature_mask = state.feature_mask
        
        # 恢复各变换的状态
        for t in self.transforms:
            if t.name in state.statistics:
                t.set_state(state.statistics[t.name])
        
        self._fitted = True


# =============================================================================
# 8. 预定义处理器工厂
# =============================================================================

def create_default_processor(
    winsorize: bool = True,
    standardize: bool = True,
    fillna: bool = True,
    feature_selection: bool = False,
    sample_weighting: bool = False,
    time_decay: bool = False,
    per_date: bool = False,
    topk: Optional[int] = None,
    decay_rate: float = 0.99,
) -> DataProcessor:
    """
    创建默认数据处理器
    
    Args:
        winsorize: 是否 winsorize
        standardize: 是否标准化
        fillna: 是否填充 NaN
        feature_selection: 是否特征选择
        sample_weighting: 是否样本加权
        time_decay: 是否时间衰减加权
        per_date: 是否按日期处理
        topk: 特征选择的 top-k
        decay_rate: 时间衰减率
    """
    transforms = []
    
    if fillna:
        transforms.append(FillNaNTransform(value=0.0))
    
    if winsorize:
        transforms.append(WinsorizeTransform(lower=0.01, upper=0.99, per_date=per_date))
    
    if standardize:
        transforms.append(StandardizeTransform(per_date=per_date))
    
    if feature_selection:
        transforms.append(FeatureSelectionTransform(method="variance" if topk is None else "topk", topk=topk))
    
    if sample_weighting:
        transforms.append(SampleWeightingTransform())
    
    if time_decay:
        transforms.append(TimeDecayWeightingTransform(decay_rate=decay_rate))
    
    return DataProcessor(transforms=transforms)


def create_lightweight_processor() -> DataProcessor:
    """创建轻量级处理器 (用于搜索阶段)"""
    return DataProcessor(transforms=[
        FillNaNTransform(value=0.0),
        WinsorizeTransform(lower=0.01, upper=0.99),
    ])


def create_weighted_processor(
    use_feature_weights: bool = True,
    use_asset_weights: bool = True,
    use_industry_weights: bool = True,
    use_time_weights: bool = True,
    time_decay: bool = True,
    decay_rate: float = 0.99,
) -> DataProcessor:
    """
    创建带样本加权的处理器
    
    Args:
        use_feature_weights: 使用特征权重
        use_asset_weights: 使用股票权重
        use_industry_weights: 使用行业权重
        use_time_weights: 使用时间权重
        time_decay: 使用时间衰减
        decay_rate: 时间衰减率
    """
    transforms = [
        FillNaNTransform(value=0.0),
        WinsorizeTransform(lower=0.01, upper=0.99),
        StandardizeTransform(),
        SampleWeightingTransform(
            use_feature_weights=use_feature_weights,
            use_asset_weights=use_asset_weights,
            use_industry_weights=use_industry_weights,
            use_time_weights=use_time_weights,
        ),
    ]
    
    if time_decay:
        transforms.append(TimeDecayWeightingTransform(decay_rate=decay_rate))
    
    return DataProcessor(transforms=transforms)


# =============================================================================
# 9. TransformRegistry - 变换注册表
# =============================================================================

class TransformRegistry:
    """
    变换注册表
    
    支持动态注册和创建 Transform 实例
    
    Example:
        # 使用装饰器注册
        @TransformRegistry.register("my_transform")
        class MyTransform(Transform):
            ...
        
        # 创建实例
        transform = TransformRegistry.create("standardize", target="y", per_date=True)
    """
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """注册变换的装饰器"""
        def decorator(transform_class: type) -> type:
            cls._registry[name] = transform_class
            return transform_class
        return decorator
    
    @classmethod
    def register_class(cls, name: str, transform_class: type) -> None:
        """显式注册变换类"""
        cls._registry[name] = transform_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Transform:
        """创建变换实例"""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown transform: {name}. Available: {available}")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_registered(cls) -> List[str]:
        """列出已注册的变换"""
        return list(cls._registry.keys())
    
    @classmethod
    def get(cls, name: str) -> type:
        """获取变换类"""
        if name not in cls._registry:
            raise ValueError(f"Unknown transform: {name}")
        return cls._registry[name]


def _register_default_transforms():
    """注册默认变换"""
    TransformRegistry.register_class("identity", IdentityTransform)
    TransformRegistry.register_class("standardize", StandardizeTransform)
    TransformRegistry.register_class("rank", RankTransform)
    TransformRegistry.register_class("winsorize", WinsorizeTransform)
    TransformRegistry.register_class("fillna", FillNaNTransform)
    TransformRegistry.register_class("dropna", DropNaNTransform)
    TransformRegistry.register_class("feature_selection", FeatureSelectionTransform)
    TransformRegistry.register_class("feature_weighting", FeatureWeightingTransform)
    TransformRegistry.register_class("sample_weighting", SampleWeightingTransform)
    TransformRegistry.register_class("time_decay_weighting", TimeDecayWeightingTransform)


# 模块加载时注册默认变换
_register_default_transforms()
