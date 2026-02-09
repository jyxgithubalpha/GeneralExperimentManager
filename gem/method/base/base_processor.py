"""
DataProcessor - 数据处理器

包含:
- BaseTransform: 变换基类
- BaseDataProcessor: 数据处理器（管理多个 Transform 的 pipeline）
- FillNaNTransform: 填充缺失值
- WinsorizeTransform: 缩尾处理
- StandardizeTransform: 标准化
- RankTransform: 排名变换
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...data.data_dataclasses import SplitView, SplitViews


@dataclass
class TransformState:
    """变换状态，用于存储 fit 时计算的统计量"""
    stats: Dict[str, Any] = field(default_factory=dict)


class BaseTransform(ABC):
    """变换基类"""
    
    def __init__(self):
        self._state: Optional[TransformState] = None
    
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


class BaseDataProcessor:
    """
    数据处理器
    
    管理多个 Transform 的 pipeline
    """
    
    def __init__(self, transforms: Optional[List[BaseTransform]] = None):
        self.transforms = transforms or []
    
    def add_transform(self, transform: BaseTransform) -> "BaseDataProcessor":
        """添加变换"""
        self.transforms.append(transform)
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "BaseDataProcessor":
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
    
    def process_views(self, views: "SplitViews") -> "SplitViews":
        """
        处理 SplitViews
        
        在 train 上 fit，然后 transform train/val/test
        """
        from ...data.data_dataclasses import SplitView, SplitViews
        
        train_keys = views.train.keys.to_numpy()[:, 0] if views.train.keys is not None else None
        val_keys = views.val.keys.to_numpy()[:, 0] if views.val.keys is not None else None
        test_keys = views.test.keys.to_numpy()[:, 0] if views.test.keys is not None else None
        
        self.fit(views.train.X, views.train.y, train_keys)
        
        train_X, train_y = self.transform(views.train.X, views.train.y, train_keys)
        val_X, val_y = self.transform(views.val.X, views.val.y, val_keys)
        test_X, test_y = self.transform(views.test.X, views.test.y, test_keys)
        
        return SplitViews(
            train=SplitView(
                X=train_X,
                y=train_y,
                keys=views.train.keys,
                feature_names=views.train.feature_names,
            ),
            val=SplitView(
                X=val_X,
                y=val_y,
                keys=views.val.keys,
                feature_names=views.val.feature_names,
            ),
            test=SplitView(
                X=test_X,
                y=test_y,
                keys=views.test.keys,
                feature_names=views.test.feature_names,
            ),
        )
