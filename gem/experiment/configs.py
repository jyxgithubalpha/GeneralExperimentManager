"""
Experiment configuration dataclasses.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..data.data_dataclasses import SplitSpec


@dataclass
class StatePolicyConfig:
    """
    State policy configuration    
    Attributes:
        mode: Policy mode
        bucket_fn: Bucket grouping function
        ema_alpha: EMA smoothing coefficient
        importance_topk: Keep only top-k features
        normalize_importance: Whether to normalize importance    
    """
    mode: str = "none"
    bucket_fn: Optional[Callable[[SplitSpec], str]] = None
    ema_alpha: float = 0.3
    importance_topk: Optional[int] = None
    normalize_importance: bool = True


@dataclass
class ResourceRequest:
    """Resource request"""
    trial_gpus: float = 1.0  # Number of GPUs per trial
    final_train_gpus: float = 1.0  # Number of GPUs for final training
    trial_cpus: float = 1.0
    final_train_cpus: float = 1.0


@dataclass
class VisualizationRenderConfig:
    show: bool = False
    interval_ms: int = 800
    dpi: int = 150


@dataclass
class ImportanceVizConfig:
    enabled: bool = True
    export_data: bool = True
    heatmap: bool = True
    animation: bool = True
    distribution: bool = True
    sort_by: str = "mean"
    normalize: str = "zscore"


@dataclass
class MetricsVizConfig:
    enabled: bool = True
    export_data: bool = True
    quarterly_summary: bool = True
    cumulative_backtest: bool = True
    relative_improvement_curve: bool = True
    overview: bool = True
    distribution: bool = True
    per_metric: bool = True
    metric_names: Optional[List[str]] = None


@dataclass
class VisualizationConfig:
    enabled: bool = False
    output_subdir: str = "visualization"
    render: VisualizationRenderConfig = field(default_factory=VisualizationRenderConfig)
    importance: ImportanceVizConfig = field(default_factory=ImportanceVizConfig)
    metrics: MetricsVizConfig = field(default_factory=MetricsVizConfig)


@dataclass
class ExperimentConfig:
    """
    Experiment configuration
    """
    name: str
    output_dir: Path
    state_policy: "StatePolicyConfig" = None  # type: ignore  # Set via Hydra
    n_trials: int = 50
    parallel_trials: int = 1
    use_ray_tune: Optional[bool] = None
    seed: int = 42
    ray_address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    use_ray: bool = False
    resource_request: Optional[ResourceRequest] = None
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.n_trials < 0:
            raise ValueError(f"n_trials must be >= 0, got {self.n_trials}")
        if self.parallel_trials <= 0:
            raise ValueError(
                f"parallel_trials must be > 0, got {self.parallel_trials}"
            )
        if self.state_policy is None:
            self.state_policy = StatePolicyConfig()
