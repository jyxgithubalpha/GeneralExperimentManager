"""
Experiment module - 实验管理
"""
from .experiment_manager import ExperimentManager
from .experiment_dataclasses import (
    ExperimentConfig,
    StatePolicyConfig,
    SplitTask,
    SplitResult,
    ResourceRequest,
    # State 相关
    BaseState,
    RollingState,
    FeatureImportanceState,
    SampleWeightState,
    TuningState,
)

__all__ = [
    "ExperimentManager",
    "ExperimentConfig",
    "StatePolicyConfig",
    "SplitTask",
    "SplitResult",
    "ResourceRequest",
    # State 相关
    "BaseState",
    "RollingState",
    "FeatureImportanceState",
    "SampleWeightState",
    "TuningState",
    "DataWeightState",
]
