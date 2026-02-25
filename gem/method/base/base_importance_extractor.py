"""
BaseImportanceExtractor
"""
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import polars as pl


class BaseImportanceExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, pl.DataFrame]:
        pass
