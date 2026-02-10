"""
Method module - 训练方法组件

包含:
- base/: 基类定义 (BaseTrainer, BaseTuner, BaseEvaluator, BaseImportanceExtractor, BaseParamSpace, Method)
- lgb/: LightGBM 实现
- method_dataclasses: 训练相关数据类
- component_factory: 组件注册器

流程:
1. 从 SplitViews 利用 train/val 计算 X/y 的阈值, 均值, std 等 (StatsCalculator)
2. 对 train/val/test 做 transform 变换 (BaseTransformPipeline)
3. 在 adapter 里把数据转化成 numpy 为基础的 ray data (RayDataAdapter)
4. 得到 numpy 数据然后变换成 lgb.dataset (LightGBMAdapter)
5. 使用 RollingState 中前面已有的超参数作为起点进行搜索训练 (LightGBMTuner)
6. 使用 ray tune 和 optuna 结合搜索
7. 使用 ray trainer 和 lgb trainer 结合来训练 (LightGBMTrainer)
8. 评估并且根据结果更新状态 (BaseMethod.update_rolling_state)
"""
from .base import (
    BaseTrainer,
    BaseEvaluator,
    BaseImportanceExtractor,
    BaseTuner,
    BaseParamSpace,
    BaseMethod,
    BaseAdapter,
    RayDataAdapter,
    BaseTransformPipeline,
    BaseTransform,
    TransformContext,
    FillNaNTransform,
    WinsorizeTransform,
    StandardizeTransform,
    RankTransform,
    FeatureWeightTransform,
    StatsCalculator,
)
from .lgb import (
    LightGBMTrainer,
    LightGBMEvaluator,
    LightGBMImportanceExtractor,
    LightGBMParamSpace,
    LightGBMTuner,
    LightGBMAdapter,
)
from .method_dataclasses import (
    TransformState,
    TransformStats,
    RayDataBundle,
    RayDataViews,
    TuneConfig,
    TrainConfig,
    TuneResult,
    FitResult,
    EvalResult,
    MethodOutput,
)
from .component_factory import ComponentRegistry

__all__ = [
    # Base classes
    "BaseTrainer",
    "BaseEvaluator",
    "BaseImportanceExtractor",
    "BaseTuner",
    "BaseParamSpace",
    "BaseMethod",
    # Adapters
    "BaseAdapter",
    "RayDataAdapter",
    # Transform
    "BaseTransformPipeline",
    "BaseTransform",
    "TransformContext",
    "FillNaNTransform",
    "WinsorizeTransform",
    "StandardizeTransform",
    "RankTransform",
    "FeatureWeightTransform",
    "StatsCalculator",
    # LightGBM implementations
    "LightGBMTrainer",
    "LightGBMEvaluator",
    "LightGBMImportanceExtractor",
    "LightGBMParamSpace",
    "LightGBMTuner",
    "LightGBMAdapter",
    # Dataclasses
    "TransformState",
    "TransformStats",
    "RayDataBundle",
    "RayDataViews",
    "TuneConfig",
    "TrainConfig",
    "TuneResult",
    "FitResult",
    "EvalResult",
    "MethodOutput",
    # Factory
    "ComponentRegistry",
]
