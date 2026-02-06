"""
Experiment module - 实验管理
"""
from .experiment_manager import ExperimentManager
from .state_dataclasses import (
    ExperimentConfig,
    StatePolicyConfig,
    StatePolicyMode,
    SplitTask,
    SplitResult,
    ResourceRequest,
)

__all__ = [
    "ExperimentManager",
    "ExperimentConfig",
    "StatePolicyConfig",
    "StatePolicyMode",
    "SplitTask",
    "SplitResult",
    "ResourceRequest",
]
