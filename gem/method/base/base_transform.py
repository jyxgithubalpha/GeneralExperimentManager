"""
DataProcessor - 数据处理器

包含:
- BaseTransform: 变换基类
- BaseTransformPipeline: 数据处理器（管理多个 Transform 的 pipeline）
- StatsCalculator: 统计量计算器 (从 train/val 计算 X/y 的阈值, 均值, std 等)
- FillNaNTransform: 填充缺失值
- WinsorizeTransform: 缩尾处理
- StandardizeTransform: 标准化
- RankTransform: 排名变换
- FeatureWeightTransform: 特征加权（利用 RollingState 中的权重）

流程:
1. 从 SplitViews 的 train/val 计算统计量 (TransformStats)
2. 使用 context (来自 RollingState) 进行特征加权
3. 对 train/val/test 应用变换
4. 返回变换后的数据和统计量
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...experiment.experiment_dataclasses import RollingState

from ...data.data_dataclasses import SplitViews, SplitView, ProcessedViews
from ..method_dataclasses import TransformState, TransformStats, RayDataBundle, RayDataViews


# Transform 上下文类型别名
TransformContext = Dict[str, Any]


class BaseTransform(ABC):
    """
    变换基类
    
    支持通过 context 接收外部参数（如 RollingState 中的特征权重）。
    子类可以通过 `self._context` 访问这些参数。
    """
    
    def __init__(self):
        self._state: Optional[TransformState] = None
        self._context: TransformContext = {}
    
    def set_context(self, context: TransformContext) -> "BaseTransform":
        """设置上下文参数（由 Pipeline 调用）"""
        self._context = context
        return self
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """获取上下文中的值"""
        return self._context.get(key, default)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "BaseTransform":
        """
        拟合变换
        
        Args:
            X: 特征矩阵
            y: 标签
            keys: 可选的键数组（如 date），用于分组计算
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用变换
        
        Returns:
            (X_transformed, y_transformed)
        """
        pass
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并变换"""
        self.fit(X, y, keys)
        return self.transform(X, y, keys)
    
    def inverse_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """逆变换（可选实现）"""
        return X, y
    
    @property
    def state(self) -> Optional[TransformState]:
        return self._state


class FillNaNTransform(BaseTransform):
    """填充缺失值"""
    
    def __init__(
        self,
        value: float = 0.0,
        method: Literal["constant", "mean", "median"] = "constant",
        target: Literal["X", "y", "both"] = "both",
    ):
        super().__init__()
        self.value = value
        self.method = method
        self.target = target
        self._fill_values_X: Optional[np.ndarray] = None
        self._fill_values_y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "FillNaNTransform":
        if self.method == "constant":
            self._fill_values_X = np.full(X.shape[1], self.value)
            self._fill_values_y = np.full(y.shape[1] if y.ndim > 1 else 1, self.value)
        elif self.method == "mean":
            self._fill_values_X = np.nanmean(X, axis=0)
            self._fill_values_y = np.nanmean(y, axis=0) if y.ndim > 1 else np.array([np.nanmean(y)])
        elif self.method == "median":
            self._fill_values_X = np.nanmedian(X, axis=0)
            self._fill_values_y = np.nanmedian(y, axis=0) if y.ndim > 1 else np.array([np.nanmedian(y)])
        
        self._state = TransformState(stats={
            "fill_values_X": self._fill_values_X,
            "fill_values_y": self._fill_values_y,
        })
        return self
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            for i in range(X_out.shape[1]):
                mask = np.isnan(X_out[:, i])
                X_out[mask, i] = self._fill_values_X[i]
        
        if self.target in ("y", "both"):
            if y_out.ndim > 1:
                for i in range(y_out.shape[1]):
                    mask = np.isnan(y_out[:, i])
                    y_out[mask, i] = self._fill_values_y[i]
            else:
                mask = np.isnan(y_out)
                y_out[mask] = self._fill_values_y[0]
        
        return X_out, y_out


class WinsorizeTransform(BaseTransform):
    """缩尾处理"""
    
    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 0.99,
        target: Literal["X", "y", "both"] = "X",
        per_date: bool = True,
    ):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.target = target
        self.per_date = per_date
        self._lower_bounds_X: Optional[np.ndarray] = None
        self._upper_bounds_X: Optional[np.ndarray] = None
        self._lower_bounds_y: Optional[np.ndarray] = None
        self._upper_bounds_y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "WinsorizeTransform":
        if not self.per_date or keys is None:
            self._lower_bounds_X = np.nanquantile(X, self.lower, axis=0)
            self._upper_bounds_X = np.nanquantile(X, self.upper, axis=0)
            if y.ndim > 1:
                self._lower_bounds_y = np.nanquantile(y, self.lower, axis=0)
                self._upper_bounds_y = np.nanquantile(y, self.upper, axis=0)
            else:
                self._lower_bounds_y = np.array([np.nanquantile(y, self.lower)])
                self._upper_bounds_y = np.array([np.nanquantile(y, self.upper)])
        
        self._state = TransformState(stats={
            "lower_bounds_X": self._lower_bounds_X,
            "upper_bounds_X": self._upper_bounds_X,
            "lower_bounds_y": self._lower_bounds_y,
            "upper_bounds_y": self._upper_bounds_y,
            "per_date": self.per_date,
        })
        return self
    
    def _winsorize_array(self, arr: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray, keys: Optional[np.ndarray]) -> np.ndarray:
        out = arr.copy()
        if self.per_date and keys is not None:
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = out[mask]
                lb = np.nanquantile(sub, self.lower, axis=0)
                ub = np.nanquantile(sub, self.upper, axis=0)
                out[mask] = np.clip(sub, lb, ub)
        else:
            out = np.clip(out, lower_bounds, upper_bounds)
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._winsorize_array(X_out, self._lower_bounds_X, self._upper_bounds_X, keys)
        
        if self.target in ("y", "both"):
            y_out = self._winsorize_array(y_out, self._lower_bounds_y, self._upper_bounds_y, keys)
        
        return X_out, y_out


class StandardizeTransform(BaseTransform):
    """标准化"""
    
    def __init__(
        self,
        target: Literal["X", "y", "both"] = "X",
        eps: float = 1e-8,
        per_date: bool = True,
    ):
        super().__init__()
        self.target = target
        self.eps = eps
        self.per_date = per_date
        self._mean_X: Optional[np.ndarray] = None
        self._std_X: Optional[np.ndarray] = None
        self._mean_y: Optional[np.ndarray] = None
        self._std_y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "StandardizeTransform":
        if not self.per_date or keys is None:
            self._mean_X = np.nanmean(X, axis=0)
            self._std_X = np.nanstd(X, axis=0)
            if y.ndim > 1:
                self._mean_y = np.nanmean(y, axis=0)
                self._std_y = np.nanstd(y, axis=0)
            else:
                self._mean_y = np.array([np.nanmean(y)])
                self._std_y = np.array([np.nanstd(y)])
        
        self._state = TransformState(stats={
            "mean_X": self._mean_X,
            "std_X": self._std_X,
            "mean_y": self._mean_y,
            "std_y": self._std_y,
            "per_date": self.per_date,
        })
        return self
    
    def _standardize_array(self, arr: np.ndarray, mean: np.ndarray, std: np.ndarray, keys: Optional[np.ndarray]) -> np.ndarray:
        out = arr.copy()
        if self.per_date and keys is not None:
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = out[mask]
                m = np.nanmean(sub, axis=0)
                s = np.nanstd(sub, axis=0) + self.eps
                out[mask] = (sub - m) / s
        else:
            out = (out - mean) / (std + self.eps)
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._standardize_array(X_out, self._mean_X, self._std_X, keys)
        
        if self.target in ("y", "both"):
            y_out = self._standardize_array(y_out, self._mean_y, self._std_y, keys)
        
        return X_out, y_out
    
    def inverse_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if not self.per_date or keys is None:
            if self.target in ("X", "both"):
                X_out = X_out * (self._std_X + self.eps) + self._mean_X
            if self.target in ("y", "both"):
                y_out = y_out * (self._std_y + self.eps) + self._mean_y
        
        return X_out, y_out


class RankTransform(BaseTransform):
    """排名变换"""
    
    def __init__(
        self,
        target: Literal["X", "y", "both"] = "X",
        per_date: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.target = target
        self.per_date = per_date
        self.normalize = normalize
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "RankTransform":
        self._state = TransformState(stats={"per_date": self.per_date, "normalize": self.normalize})
        return self
    
    def _rank_array(self, arr: np.ndarray, keys: Optional[np.ndarray]) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float32)
        
        def rank_1d(x: np.ndarray) -> np.ndarray:
            ranks = np.argsort(np.argsort(x)).astype(np.float32)
            if self.normalize:
                ranks = ranks / (len(x) - 1) if len(x) > 1 else ranks
            return ranks
        
        if self.per_date and keys is not None:
            unique_keys = np.unique(keys)
            for key in unique_keys:
                mask = keys == key
                sub = arr[mask]
                if sub.ndim > 1:
                    for i in range(sub.shape[1]):
                        out[mask, i] = rank_1d(sub[:, i])
                else:
                    out[mask] = rank_1d(sub)
        else:
            if arr.ndim > 1:
                for i in range(arr.shape[1]):
                    out[:, i] = rank_1d(arr[:, i])
            else:
                out = rank_1d(arr)
        
        return out
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        X_out = X.copy()
        y_out = y.copy()
        
        if self.target in ("X", "both"):
            X_out = self._rank_array(X_out, keys)
        
        if self.target in ("y", "both"):
            y_out = self._rank_array(y_out, keys)
        
        return X_out, y_out


class FeatureWeightTransform(BaseTransform):
    """
    特征加权变换
    
    从 context 中获取 feature_weights，对 X 进行加权。
    支持多种加权方式:
    - multiply: X * weights
    - softmax: X * softmax(weights)
    - select_topk: 只保留 top-k 重要特征
    
    context 中期望的 key:
    - feature_weights: np.ndarray - 特征权重向量
    """
    
    def __init__(
        self,
        method: Literal["multiply", "softmax", "select_topk"] = "multiply",
        topk: Optional[int] = None,
        temperature: float = 1.0,
        fallback: Literal["uniform", "ones", "skip"] = "ones",
        context_key: str = "feature_weights",
    ):
        """
        Args:
            method: 加权方式
            topk: 保留的 top-k 特征数量 (仅用于 select_topk)
            temperature: softmax 温度 (仅用于 softmax)
            fallback: 当 context 中没有 feature_weights 时的处理方式
                - uniform: 均匀权重 (1/n_features)
                - ones: 全 1 权重 (不变)
                - skip: 跳过变换
            context_key: 从 context 中读取权重的 key
        """
        super().__init__()
        self.method = method
        self.topk = topk
        self.temperature = temperature
        self.fallback = fallback
        self.context_key = context_key
        self._weights: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "FeatureWeightTransform":
        n_features = X.shape[1]
        
        # 从 context 获取权重
        raw_weights = self.get_context_value(self.context_key)
        
        if raw_weights is None:
            raw_weights = self._get_fallback_weights(n_features)
        elif len(raw_weights) != n_features:
            raw_weights = self._get_fallback_weights(n_features)
        else:
            raw_weights = np.array(raw_weights, dtype=np.float32)
        
        # 处理权重
        if self.method == "multiply":
            self._weights = raw_weights
        elif self.method == "softmax":
            exp_w = np.exp(raw_weights / self.temperature)
            self._weights = exp_w / exp_w.sum()
        elif self.method == "select_topk":
            k = self.topk or n_features
            k = min(k, n_features)
            self._selected_indices = np.argsort(raw_weights)[-k:]
            self._weights = None
        
        self._state = TransformState(stats={
            "weights": self._weights,
            "selected_indices": self._selected_indices,
            "method": self.method,
        })
        return self
    
    def _get_fallback_weights(self, n_features: int) -> Optional[np.ndarray]:
        if self.fallback == "uniform":
            return np.full(n_features, 1.0 / n_features, dtype=np.float32)
        elif self.fallback == "ones":
            return np.ones(n_features, dtype=np.float32)
        else:  # skip
            return None
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # skip 模式且没有权重时跳过
        if self.fallback == "skip" and self._weights is None and self._selected_indices is None:
            return X, y
        
        X_out = X.copy()
        
        if self.method == "select_topk" and self._selected_indices is not None:
            X_out = X_out[:, self._selected_indices]
        elif self._weights is not None:
            X_out = X_out * self._weights
        
        return X_out, y
    
    @property
    def selected_feature_indices(self) -> Optional[np.ndarray]:
        """获取选中的特征索引 (select_topk 模式)"""
        return self._selected_indices


class BaseTransformPipeline:
    """
    数据处理器
    
    管理多个 Transform 的 pipeline，支持通过 context 传递外部参数。
    """
    
    def __init__(self, transforms: Optional[List[BaseTransform]] = None):
        self.transforms = transforms or []
        self._context: TransformContext = {}
    
    def set_context(self, context: TransformContext) -> "BaseTransformPipeline":
        """设置上下文，传递给所有 transform"""
        self._context = context
        for transform in self.transforms:
            transform.set_context(context)
        return self
    
    def add_transform(self, transform: BaseTransform) -> "BaseTransformPipeline":
        """添加变换"""
        transform.set_context(self._context)
        self.transforms.append(transform)
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "BaseTransformPipeline":
        """拟合所有变换"""
        X_curr, y_curr = X.copy(), y.copy()
        for transform in self.transforms:
            transform.fit(X_curr, y_curr, keys)
            X_curr, y_curr = transform.transform(X_curr, y_curr, keys)
        return self
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """应用所有变换"""
        X_curr, y_curr = X.copy(), y.copy()
        for transform in self.transforms:
            X_curr, y_curr = transform.transform(X_curr, y_curr, keys)
        return X_curr, y_curr
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并变换"""
        self.fit(X, y, keys)
        return self.transform(X, y, keys)
    
    def inverse_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """逆变换（反向遍历）"""
        X_curr, y_curr = X.copy(), y.copy()
        for transform in reversed(self.transforms):
            X_curr, y_curr = transform.inverse_transform(X_curr, y_curr, keys)
        return X_curr, y_curr
    
    def process_views(
        self,
        views: "SplitViews",
        context: Optional[TransformContext] = None,
    ) -> "SplitViews":
        """
        处理 SplitViews
        
        Args:
            views: 输入视图
            context: 可选的上下文参数，可从 RollingState.to_transform_context() 获取
        
        在 train 上 fit，然后 transform train/val/test
        """
        from ...data.data_dataclasses import SplitView, SplitViews
        
        # 设置 context
        if context is not None:
            self.set_context(context)
        
        train_keys = views.train.keys.to_numpy()[:, 0] if views.train.keys is not None else None
        val_keys = views.val.keys.to_numpy()[:, 0] if views.val.keys is not None else None
        test_keys = views.test.keys.to_numpy()[:, 0] if views.test.keys is not None else None
        
        self.fit(views.train.X, views.train.y, train_keys)
        
        train_X, train_y = self.transform(views.train.X, views.train.y, train_keys)
        val_X, val_y = self.transform(views.val.X, views.val.y, val_keys)
        test_X, test_y = self.transform(views.test.X, views.test.y, test_keys)
        
        # 处理 select_topk 的 feature_names 更新
        feature_names = views.train.feature_names
        for transform in self.transforms:
            if isinstance(transform, FeatureWeightTransform) and transform.selected_feature_indices is not None:
                if feature_names is not None:
                    feature_names = [feature_names[i] for i in transform.selected_feature_indices]
        
        return SplitViews(
            train=SplitView(
                indices=views.train.indices,
                X=train_X,
                y=train_y,
                keys=views.train.keys,
                feature_names=feature_names,
                label_names=views.train.label_names,
                extra=views.train.extra,
            ),
            val=SplitView(
                indices=views.val.indices,
                X=val_X,
                y=val_y,
                keys=views.val.keys,
                feature_names=feature_names,
                label_names=views.val.label_names,
                extra=views.val.extra,
            ),
            test=SplitView(
                indices=views.test.indices,
                X=test_X,
                y=test_y,
                keys=views.test.keys,
                feature_names=feature_names,
                label_names=views.test.label_names,
                extra=views.test.extra,
            ),
            split_spec=views.split_spec,
        )
    
    def fit_transform(
        self,
        views: "SplitViews",
        rolling_state: Optional["RollingState"] = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> Tuple["ProcessedViews", TransformStats]:
        """
        完整的 fit + transform 流程，返回 ProcessedViews 和 TransformStats
        
        Args:
            views: 输入视图
            rolling_state: 滚动状态 (用于获取 context)
            lower_quantile: 下分位数阈值
            upper_quantile: 上分位数阈值
            
        Returns:
            (ProcessedViews, TransformStats)
        """
        # 1. 从 RollingState 获取 context
        context = {}
        if rolling_state is not None:
            context = rolling_state.to_transform_context()
        
        # 2. 计算统计量
        stats = StatsCalculator.compute(
            X_train=views.train.X,
            y_train=views.train.y,
            X_val=views.val.X,
            y_val=views.val.y,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        
        # 3. 处理视图
        transformed_views = self.process_views(views, context)
        
        # 4. 构建 ProcessedViews
        processed = ProcessedViews(
            train=transformed_views.train,
            val=transformed_views.val,
            test=transformed_views.test,
            split_spec=views.split_spec,
            transform_state=TransformState(
                stats=self._collect_all_stats(),
                transform_stats=stats,
            ),
        )
        
        return processed, stats
    
    def _collect_all_stats(self) -> Dict[str, Any]:
        """收集所有 transform 的统计量"""
        all_stats = {}
        for i, transform in enumerate(self.transforms):
            if transform.state is not None:
                all_stats[f"transform_{i}_{type(transform).__name__}"] = transform.state.stats
        return all_stats
    
    def to_ray_data_views(
        self,
        views: "SplitViews",
        rolling_state: Optional["RollingState"] = None,
        include_sample_weight: bool = True,
    ) -> Tuple[RayDataViews, TransformStats]:
        """
        完整流程: transform -> 转换为 RayDataViews
        
        Args:
            views: 输入视图
            rolling_state: 滚动状态
            include_sample_weight: 是否包含样本权重
            
        Returns:
            (RayDataViews, TransformStats)
        """
        # 1. Fit transform
        processed, stats = self.fit_transform(views, rolling_state)
        
        # 2. 获取样本权重 (如果有)
        sample_weight_train = None
        sample_weight_val = None
        sample_weight_test = None
        
        if include_sample_weight and rolling_state is not None:
            from ...experiment.experiment_dataclasses import SampleWeightState
            weight_state = rolling_state.get_state(SampleWeightState)
            if weight_state is not None and hasattr(weight_state, 'get_weights_for_view'):
                sample_weight_train = weight_state.get_weights_for_view(processed.train)
                sample_weight_val = weight_state.get_weights_for_view(processed.val)
                sample_weight_test = weight_state.get_weights_for_view(processed.test)
        
        # 3. 构建 RayDataBundle
        def _to_bundle(view: SplitView, weight: Optional[np.ndarray]) -> RayDataBundle:
            keys_arr = None
            if view.keys is not None:
                keys_arr = view.keys.to_numpy()
            return RayDataBundle(
                X=view.X,
                y=view.y,
                keys=keys_arr,
                sample_weight=weight,
                feature_names=view.feature_names,
                label_names=view.label_names,
            )
        
        ray_views = RayDataViews(
            train=_to_bundle(processed.train, sample_weight_train),
            val=_to_bundle(processed.val, sample_weight_val),
            test=_to_bundle(processed.test, sample_weight_test),
            transform_state=processed.transform_state,
            transform_stats=stats,
        )
        
        return ray_views, stats


class StatsCalculator:
    """
    统计量计算器
    
    从 train/val 数据计算 X 和 y 的统计量:
    - 均值 (mean)
    - 标准差 (std)
    - 分位数 (quantiles)
    - 中位数 (median)
    """
    
    @staticmethod
    def compute(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        use_combined: bool = True,
    ) -> TransformStats:
        """
        计算统计量
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征 (可选)
            y_val: 验证集标签 (可选)
            lower_quantile: 下分位数
            upper_quantile: 上分位数
            use_combined: 是否合并 train 和 val 计算
            
        Returns:
            TransformStats
        """
        # 合并数据 (如果需要)
        if use_combined and X_val is not None:
            X = np.vstack([X_train, X_val])
            y = np.concatenate([y_train.ravel(), y_val.ravel()])
        else:
            X = X_train
            y = y_train.ravel() if y_train.ndim > 1 else y_train
        
        # 计算 X 统计量
        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)
        X_lower = np.nanquantile(X, lower_quantile, axis=0)
        X_upper = np.nanquantile(X, upper_quantile, axis=0)
        X_median = np.nanmedian(X, axis=0)
        
        # 计算 y 统计量
        y_mean = np.array([np.nanmean(y)])
        y_std = np.array([np.nanstd(y)])
        y_lower = np.array([np.nanquantile(y, lower_quantile)])
        y_upper = np.array([np.nanquantile(y, upper_quantile)])
        y_median = np.array([np.nanmedian(y)])
        
        return TransformStats(
            X_mean=X_mean,
            X_std=X_std,
            X_lower_quantile=X_lower,
            X_upper_quantile=X_upper,
            X_median=X_median,
            y_mean=y_mean,
            y_std=y_std,
            y_lower_quantile=y_lower,
            y_upper_quantile=y_upper,
            y_median=y_median,
        )
    
    @staticmethod
    def compute_per_date(
        X: np.ndarray,
        y: np.ndarray,
        keys: np.ndarray,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> Dict[int, TransformStats]:
        """
        按日期分组计算统计量
        
        Args:
            X: 特征矩阵
            y: 标签
            keys: 日期键
            lower_quantile: 下分位数
            upper_quantile: 上分位数
            
        Returns:
            {date: TransformStats}
        """
        stats_by_date = {}
        unique_dates = np.unique(keys)
        
        for date in unique_dates:
            mask = keys == date
            X_sub = X[mask]
            y_sub = y[mask]
            
            stats_by_date[int(date)] = StatsCalculator.compute(
                X_train=X_sub,
                y_train=y_sub,
                use_combined=False,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
        
        return stats_by_date
