"""
Experiment module - 实验管理

核心组件:
- ExperimentManager: 实验编排入口
- RunContext: 运行上下文
- SplitRunner: 单 split 执行器
- LocalExecutor/RayExecutor: 执行后端
"""
from .experiment_manager import ExperimentManager
from .experiment_dataclasses import (
    ExperimentConfig,
    VisualizationConfig,
    StatePolicyConfig,
    SplitTask,
    SplitResult,
    ResourceRequest,
    RollingState,
    FeatureImportanceState,
)
from .run_context import RunContext
from .split_runner import SplitRunner
from .executor import LocalExecutor, RayExecutor
from .task_dag import DynamicTaskDAG, DagSubmission

__all__ = [
    "ExperimentManager",
    "ExperimentConfig",
    "VisualizationConfig",
    "StatePolicyConfig",
    "SplitTask",
    "SplitResult",
    "ResourceRequest",
    "RollingState",
    "FeatureImportanceState",
    "RunContext",
    "SplitRunner",
    "LocalExecutor",
    "RayExecutor",
    "DynamicTaskDAG",
    "DagSubmission",
]
