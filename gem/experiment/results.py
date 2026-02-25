"""
Split task and result dataclasses.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import polars as pl

from ..data.data_dataclasses import SplitSpec
from .configs import ResourceRequest

if TYPE_CHECKING:
    from ..method.method_dataclasses import StateDelta


@dataclass
class SplitTask:
    """Split task - built by ExperimentManager."""
    split_id: int
    splitspec: SplitSpec
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)


@dataclass
class SplitResult:
    """Split result - result of a single split training."""
    split_id: int
    importance_vector: Optional[np.ndarray] = None
    feature_names_hash: Optional[str] = None
    industry_delta: Optional[Dict[str, float]] = None
    metrics: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_objective: Optional[float] = None
    state_delta: Optional["StateDelta"] = None
    skipped: bool = False
    failed: bool = False
    skip_reason: Optional[str] = None
    error_message: Optional[str] = None
    error_trace_path: Optional[str] = None
    metric_series_rows: Optional[List[Dict[str, Any]]] = None
    test_predictions: Optional[np.ndarray] = None
    test_keys: Optional["pl.DataFrame"] = None
    test_extra: Optional["pl.DataFrame"] = None
