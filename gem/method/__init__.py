"""
Method module - 训练方法组件

包含:
- base/: 基类定义 (BaseTrainer, BaseTuner, BaseEvaluator, BaseImportanceExtractor, BaseParamSpace, Method)
- lgb/: LightGBM 实现
- xgb/: XGBoost 实现
- catboost/: CatBoost 实现
- mlp/: MLP 实现
- bayesian/: 贝叶斯方法实�?
- training_dataclasses: 训练相关数据�?
"""
from .base import (
    BaseTrainer,
    BaseEvaluator,
    BaseImportanceExtractor,
    BaseTuner,
    BaseParamSpace,
    Method,
)
from .lgb import (
    LightGBMTrainer,
    LightGBMEvaluator,
    LightGBMImportanceExtractor,
    LightGBMParamSpace,
    LightGBMTuner,
)
from .xgb import (
    XGBoostTrainer,
    XGBoostEvaluator,
    XGBoostImportanceExtractor,
    XGBoostParamSpace,
    XGBoostTuner,
)
from .catboost import (
    CatBoostTrainer,
    CatBoostEvaluator,
    CatBoostImportanceExtractor,
    CatBoostParamSpace,
    CatBoostTuner,
)
from .mlp import (
    MLPTrainer,
    MLPEvaluator,
    MLPImportanceExtractor,
    MLPParamSpace,
    MLPTuner,
)
from .bayesian import (
    BayesianTrainer,
    BayesianEvaluator,
    BayesianImportanceExtractor,
    BayesianParamSpace,
    BayesianTuner,
)
from .training_dataclasses import (
    TrainConfig,
    FitResult,
    EvalResult,
    MethodOutput,
)

__all__ = [
    # Base classes
    "BaseTrainer",
    "BaseEvaluator",
    "BaseImportanceExtractor",
    "BaseTuner",
    "BaseParamSpace",
    "Method",
    # LightGBM implementations
    "LightGBMTrainer",
    "LightGBMEvaluator",
    "LightGBMImportanceExtractor",
    "LightGBMParamSpace",
    "LightGBMTuner",
    # XGBoost implementations
    "XGBoostTrainer",
    "XGBoostEvaluator",
    "XGBoostImportanceExtractor",
    "XGBoostParamSpace",
    "XGBoostTuner",
    # CatBoost implementations
    "CatBoostTrainer",
    "CatBoostEvaluator",
    "CatBoostImportanceExtractor",
    "CatBoostParamSpace",
    "CatBoostTuner",
    # MLP implementations
    "MLPTrainer",
    "MLPEvaluator",
    "MLPImportanceExtractor",
    "MLPParamSpace",
    "MLPTuner",
    # Bayesian implementations
    "BayesianTrainer",
    "BayesianEvaluator",
    "BayesianImportanceExtractor",
    "BayesianParamSpace",
    "BayesianTuner",
    # Dataclasses
    "TrainConfig",
    "FitResult",
    "EvalResult",
    "MethodOutput",
]
