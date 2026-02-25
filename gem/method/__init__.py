"""
Method module public exports.

Supports:
- GBDT: LightGBM, XGBoost, CatBoost
- sklearn: Various regression models
- PyTorch: MLP, Transformer with NNI NAS support

Core components:
- UnifiedTuner: Framework-agnostic hyperparameter/architecture tuner
- TunerBackend: Optuna, RayTune, NNI backends
- BaseSearchSpace: Parameter/architecture search spaces
"""

from .base import (
    BaseAdapter,
    BaseEvaluator,
    BaseImportanceExtractor,
    BaseMethod,
    BaseTrainer,
    BaseTransform,
    BaseTransformPipeline,
    FeatureWeightTransform,
    FillNaNTransform,
    FittedTransformPipeline,
    MADStandardizeTransform,
    MethodComponents,
    MinSampleFilterTransform,
    RankTransform,
    RayDataAdapter,
    StandardizeTransform,
    StatsCalculator,
    TransformContext,
    WinsorizeTransform,
    BaseSearchSpace,
    TunerBackend,
    OptunaBackend,
    RayTuneBackend,
    UnifiedTuner,
)
from .base import RegressionEvaluator, PortfolioBacktestCalculator, PortfolioBacktestConfig
from .gbdt import (
    LightGBMSpace,
    XGBoostSpace,
    CatBoostSpace,
    GBDTImportanceExtractor,
    LightGBMAdapter,
    XGBoostAdapter,
    CatBoostAdapter,
    LightGBMTrainer,
    XGBoostTrainer,
    CatBoostTrainer,
)
from .method_dataclasses import (
    EvalResult,
    FitResult,
    MethodOutput,
    RayDataBundle,
    RayDataViews,
    StateDelta,
    TrainConfig,
    TransformState,
    TransformStats,
    TuneResult,
)
from .method_factory import MethodFactory

__all__ = [
    # Base classes
    "BaseAdapter",
    "BaseEvaluator",
    "BaseImportanceExtractor",
    "BaseMethod",
    "BaseTrainer",
    "MethodComponents",
    # Adapters
    "RayDataAdapter",
    # Tuning (new unified system)
    "BaseSearchSpace",
    "TunerBackend",
    "OptunaBackend",
    "RayTuneBackend",
    "UnifiedTuner",
    # Transforms
    "BaseTransform",
    "BaseTransformPipeline",
    "FittedTransformPipeline",
    "TransformContext",
    "FeatureWeightTransform",
    "FillNaNTransform",
    "MADStandardizeTransform",
    "MinSampleFilterTransform",
    "RankTransform",
    "StandardizeTransform",
    "StatsCalculator",
    "WinsorizeTransform",
    # GBDT
    "LightGBMSpace",
    "XGBoostSpace",
    "CatBoostSpace",
    "GBDTImportanceExtractor",
    "LightGBMAdapter",
    "XGBoostAdapter",
    "CatBoostAdapter",
    "LightGBMTrainer",
    "XGBoostTrainer",
    "CatBoostTrainer",
    # Evaluators
    "RegressionEvaluator",
    "PortfolioBacktestCalculator",
    "PortfolioBacktestConfig",
    # Dataclasses
    "EvalResult",
    "FitResult",
    "MethodOutput",
    "RayDataBundle",
    "RayDataViews",
    "StateDelta",
    "TrainConfig",
    "TransformState",
    "TransformStats",
    "TuneResult",
    # Factory
    "MethodFactory",
]
