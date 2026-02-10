"""
Method 相关数据类

包含:
- TransformStats: 变换统计量 (阈值, 均值, std等)
- TransformState: 变换状态
- RayDataBundle: Ray Data 数据包
- RayDataViews: Ray Data 视图集合
- TuneConfig: 搜索配置
- TrainConfig: 训练配置
- TuneResult: 搜索结果
- FitResult: 训练结果
- EvalResult: 评估结果
- MethodOutput: Method 完整输出
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


# =============================================================================
# Transform 相关
# =============================================================================

@dataclass
class TransformStats:
    """
    变换统计量 - 从 train/val 计算得到
    
    包含 X 和 y 的:
    - 阈值 (quantiles)
    - 均值 (mean)
    - 标准差 (std)
    - 其他自定义统计量
    """
    # X 统计量
    X_mean: Optional[np.ndarray] = None
    X_std: Optional[np.ndarray] = None
    X_lower_quantile: Optional[np.ndarray] = None
    X_upper_quantile: Optional[np.ndarray] = None
    X_median: Optional[np.ndarray] = None
    
    # y 统计量
    y_mean: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None
    y_lower_quantile: Optional[np.ndarray] = None
    y_upper_quantile: Optional[np.ndarray] = None
    y_median: Optional[np.ndarray] = None
    
    # 自定义统计量
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "X_lower_quantile": self.X_lower_quantile,
            "X_upper_quantile": self.X_upper_quantile,
            "X_median": self.X_median,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "y_lower_quantile": self.y_lower_quantile,
            "y_upper_quantile": self.y_upper_quantile,
            "y_median": self.y_median,
            **self.custom,
        }


@dataclass
class TransformState:
    """
    变换状态 - 用于存储 fit 时计算的统计量
    
    支持链式 Transform 的状态存储
    """
    stats: Dict[str, Any] = field(default_factory=dict)
    transform_stats: Optional[TransformStats] = None

# =============================================================================
# Ray Data 相关
# =============================================================================

@dataclass
class RayDataBundle:
    """
    Ray Data 数据包 - 用于 Ray 分布式训练
    
    存储从 pl.DataFrame 转换后的 numpy 数组和 ray.data.Dataset
    """
    # Numpy 数据
    X: np.ndarray
    y: np.ndarray
    keys: Optional[np.ndarray] = None
    sample_weight: Optional[np.ndarray] = None
    
    # 元信息
    feature_names: Optional[List[str]] = None
    label_names: Optional[List[str]] = None
    n_samples: int = 0
    n_features: int = 0
    
    # Ray Data (延迟创建)
    _ray_dataset: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1] if self.X.ndim > 1 else 1
    
    def to_ray_dataset(self, include_weight: bool = True) -> Any:
        """
        转换为 ray.data.Dataset
        
        Returns:
            ray.data.Dataset
        """
        try:
            import ray.data
        except ImportError:
            raise ImportError("ray[data] is required. Install with: pip install 'ray[data]'")
        
        if self._ray_dataset is not None:
            return self._ray_dataset
        
        # 构建数据字典
        data_dict = {
            "X": self.X,
            "y": self.y,
        }
        if self.keys is not None:
            data_dict["keys"] = self.keys
        if include_weight and self.sample_weight is not None:
            data_dict["sample_weight"] = self.sample_weight
        
        self._ray_dataset = ray.data.from_numpy(data_dict)
        return self._ray_dataset


@dataclass
class RayDataViews:
    """
    Ray Data 视图集合 - train/val/test
    """
    train: RayDataBundle
    val: RayDataBundle
    test: RayDataBundle
    transform_state: Optional[TransformState] = None
    transform_stats: Optional[TransformStats] = None


# =============================================================================
# Training 配置相关
# =============================================================================

@dataclass
class TuneConfig:
    """
    超参搜索配置
    
    Attributes:
        n_trials: 搜索次数
        timeout: 超时时间 (秒)
        target_metric: 目标指标
        direction: 优化方向 (maximize/minimize)
        use_ray_tune: 是否使用 Ray Tune
        use_optuna: 是否使用 Optuna
        parallel_trials: 并行 trial 数
        use_warm_start: 是否使用历史超参作为起点
        shrink_ratio: 搜索空间收缩比例
    """
    n_trials: int = 50
    timeout: Optional[int] = None
    target_metric: str = "pearsonr_ic"
    direction: str = "maximize"
    use_ray_tune: bool = True
    use_optuna: bool = True
    parallel_trials: int = 1
    use_warm_start: bool = True
    shrink_ratio: float = 0.5
    seed: int = 42


@dataclass
class TrainConfig:
    """
    训练配置
    
    Attributes:
        params: 模型超参数
        num_boost_round: 最大迭代次数
        early_stopping_rounds: 早停轮数
        feval_names: 评估指标名列表
        objective_name: 目标函数名
        seed: 随机种子
        verbose_eval: 日志打印频率
        use_ray_trainer: 是否使用 Ray Trainer
    """
    params: Dict[str, Any]
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    feval_names: List[str] = field(default_factory=lambda: ["pearsonr_ic"])
    objective_name: str = "regression"
    seed: int = 42
    verbose_eval: int = 100
    use_ray_trainer: bool = False
    
    def with_params(self, **kwargs) -> "TrainConfig":
        """返回新配置，覆盖部分参数"""
        new_params = {**self.params, **kwargs}
        return TrainConfig(
            params=new_params,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            feval_names=self.feval_names,
            objective_name=self.objective_name,
            seed=self.seed,
            verbose_eval=self.verbose_eval,
            use_ray_trainer=self.use_ray_trainer,
        )
    
    def for_tuning(self, params: Dict[str, Any], seed: int) -> "TrainConfig":
        """创建用于调参的轻量配置"""
        return TrainConfig(
            params=params,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            feval_names=self.feval_names[:1],  # 只用第一个指标
            objective_name=self.objective_name,
            seed=seed,
            verbose_eval=0,
            use_ray_trainer=False,  # 调参时不用 Ray Trainer
        )


# =============================================================================
# 训练结果相关
# =============================================================================

@dataclass
class TuneResult:
    """
    超参搜索结果
    
    Attributes:
        best_params: 最佳超参数
        best_value: 最佳目标值
        n_trials: 完成的 trial 数
        all_trials: 所有 trial 结果
        search_time: 搜索耗时
        warm_start_used: 是否使用了热启动
        shrunk_space_used: 是否使用了收缩空间
    """
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    all_trials: Optional[List[Dict[str, Any]]] = None
    search_time: float = 0.0
    warm_start_used: bool = False
    shrunk_space_used: bool = False


@dataclass
class FitResult:
    """
    训练结果
    
    Attributes:
        model: 训练好的模型
        evals_result: 评估结果历史
        best_iteration: 最佳迭代次数
        params: 使用的超参数
        seed: 随机种子
        train_time: 训练耗时
        feature_importance: 特征重要性 DataFrame
        checkpoint_path: Ray 检查点路径 (如果使用 Ray Trainer)
    """
    model: Any  # lgb.Booster or other
    evals_result: Dict[str, Dict[str, List[float]]]
    best_iteration: int
    params: Dict[str, Any]
    seed: int
    train_time: float = 0.0
    feature_importance: Optional[pl.DataFrame] = None
    checkpoint_path: Optional[Path] = None


@dataclass 
class EvalResult:
    """
    评估结果
    
    Attributes:
        metrics: 指标字典
        series: 时间序列指标
        predictions: 预测值
        mode: 评估模式 (train/val/test)
    """
    metrics: Dict[str, float]
    series: Dict[str, pl.Series] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    mode: str = ""


@dataclass
class MethodOutput:
    """
    Method 的完整输出
    
    Attributes:
        best_params: 最佳超参数 (来自 tuner)
        metrics_eval: 评估阶段指标 {mode: EvalResult}
        importance_vector: 特征重要性向量 (与当前 feature_names 对齐)
        feature_names_hash: 特征名哈希 (防错)
        tune_result: 搜索结果
        fit_result: 训练结果
        transform_stats: 变换统计量
        state_update: 状态更新增量
        model_artifacts: 模型产物路径
    """
    best_params: Dict[str, Any]
    metrics_eval: Dict[str, EvalResult]
    importance_vector: np.ndarray
    feature_names_hash: str
    tune_result: Optional[TuneResult] = None
    fit_result: Optional[FitResult] = None
    transform_stats: Optional[TransformStats] = None
    state_update: Optional[Dict[str, Any]] = None
    model_artifacts: Optional[Dict[str, Path]] = None
    
    @property
    def metrics_search(self) -> Optional[Dict[str, float]]:
        """兼容旧 API"""
        if self.tune_result is None:
            return None
        return {"best_value": self.tune_result.best_value}
    
    def get_state_delta(self) -> Dict[str, Any]:
        """
        获取用于更新 RollingState 的增量
        
        Returns:
            包含 importance_vector, best_params, best_objective 的字典
        """
        delta = {
            "importance_vector": self.importance_vector,
            "feature_names_hash": self.feature_names_hash,
            "best_params": self.best_params,
        }
        if self.tune_result is not None:
            delta["best_objective"] = self.tune_result.best_value
        return delta
