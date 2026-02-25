"""
Base components for Method module.

Contains:
- BaseTrainer: Trainer base class
- BaseEvaluator: Evaluator base class
- BaseImportanceExtractor: Feature importance extractor base class
- BaseMethod: Unified training interface
- BaseAdapter: Data adapter base class
- RayDataAdapter: Ray Data adapter
- Tuning: UnifiedTuner, TunerBackend, BaseSearchSpace
- Transforms: Pipeline and transform implementations
"""
from .base_trainer import BaseTrainer
from .base_evaluator import BaseEvaluator
from .base_importance_extractor import BaseImportanceExtractor
from .base_method import BaseMethod, MethodComponents
from .base_adapter import BaseAdapter, RayDataAdapter
from .evaluators import (
    RegressionEvaluator,
    PortfolioBacktestCalculator,
    PortfolioBacktestConfig,
)

from .tuning import (
    BaseSearchSpace,
    TunerBackend,
    OptunaBackend,
    RayTuneBackend,
    UnifiedTuner,
)

from .transforms import (
    BaseTransform,
    BaseTransformPipeline,
    FittedTransformPipeline,
    TransformContext,
    FillNaNTransform,
    WinsorizeTransform,
    StandardizeTransform,
    MADStandardizeTransform,
    MinSampleFilterTransform,
    RankTransform,
    FeatureWeightTransform,
    StatsCalculator,
    extract_date_keys,
)

__all__ = [
    # Base classes
    "BaseTrainer",
    "BaseEvaluator",
    "BaseImportanceExtractor",
    "BaseMethod",
    "MethodComponents",
    # Adapters
    "BaseAdapter",
    "RayDataAdapter",
    # Evaluators
    "RegressionEvaluator",
    "PortfolioBacktestCalculator",
    "PortfolioBacktestConfig",
    # Tuning
    "BaseSearchSpace",
    "TunerBackend",
    "OptunaBackend",
    "RayTuneBackend",
    "UnifiedTuner",
    # Transform
    "BaseTransform",
    "BaseTransformPipeline",
    "FittedTransformPipeline",
    "TransformContext",
    "FillNaNTransform",
    "WinsorizeTransform",
    "StandardizeTransform",
    "MADStandardizeTransform",
    "MinSampleFilterTransform",
    "RankTransform",
    "FeatureWeightTransform",
    "StatsCalculator",
    "extract_date_keys",
]
