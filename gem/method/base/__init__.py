"""
Base components for Method module

包含:
- BaseTrainer: 训练器基类
- BaseEvaluator: 评估器基类
- BaseImportanceExtractor: 特征重要性提取器基类
- BaseTuner: 超参调优器基类
- BaseParamSpace: 参数空间基类
- BaseMethod: 统一训练接口
- BaseAdapter: 数据适配器基类
- RayDataAdapter: Ray Data 适配器
- BaseTransformPipeline: 变换 pipeline
- StatsCalculator: 统计量计算器
- Transform 实现类
"""
from .base_trainer import BaseTrainer
from .base_evaluator import BaseEvaluator
from .base_importance_extractor import BaseImportanceExtractor
from .base_tuner import BaseTuner
from .base_param_space import BaseParamSpace
from .base_method import BaseMethod, MethodComponents
from .base_adapter import BaseAdapter, RayDataAdapter
from .base_transform import (
    BaseTransform,
    BaseTransformPipeline,
    FittedTransformPipeline,
    TransformContext,
    FillNaNTransform,
    WinsorizeTransform,
    StandardizeTransform,
    RankTransform,
    FeatureWeightTransform,
    StatsCalculator,
)

__all__ = [
    # Base classes
    "BaseTrainer",
    "BaseEvaluator",
    "BaseImportanceExtractor",
    "BaseTuner",
    "BaseParamSpace",
    "BaseMethod",
    "MethodComponents",
    # Adapters
    "BaseAdapter",
    "RayDataAdapter",
    # Transform
    "BaseTransform",
    "BaseTransformPipeline",
    "FittedTransformPipeline",
    "TransformContext",
    "FillNaNTransform",
    "WinsorizeTransform",
    "StandardizeTransform",
    "RankTransform",
    "FeatureWeightTransform",
    "StatsCalculator",
]
