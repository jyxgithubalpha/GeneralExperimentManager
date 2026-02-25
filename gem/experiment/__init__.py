"""Experiment module - 实验管理"""

from .configs import ExperimentConfig, StatePolicyConfig, ResourceRequest
from .results import SplitTask, SplitResult
from .states import RollingState, FeatureImportanceState
from .report import ReportGenerator
from .run_context import RunContext
from .task_dag import DynamicTaskDAG, DagSubmission


def __getattr__(name):
    """Lazy imports to avoid circular dependencies."""
    if name == "ExperimentManager":
        from .experiment_manager import ExperimentManager
        return ExperimentManager
    if name == "SplitRunner":
        from .split_runner import SplitRunner
        return SplitRunner
    if name == "LocalExecutor":
        from .executor import LocalExecutor
        return LocalExecutor
    if name == "RayExecutor":
        from .executor import RayExecutor
        return RayExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExperimentManager",
    "ExperimentConfig",
    "StatePolicyConfig",
    "ResourceRequest",
    "SplitTask",
    "SplitResult",
    "RollingState",
    "FeatureImportanceState",
    "ReportGenerator",
    "RunContext",
    "SplitRunner",
    "LocalExecutor",
    "RayExecutor",
    "DynamicTaskDAG",
    "DagSubmission",
]
