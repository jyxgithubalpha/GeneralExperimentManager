"""
Feval适配器 - 将Metric适配为LightGBM feval
"""



from typing import Callable, Dict, List, Tuple

import lightgbm as lgb
import numpy as np

from ..data.data_dataclasses import SplitView
from .metrics import Metric, MetricRegistry


class FevalAdapter:
    """将Metric适配为LightGBM feval格式"""
    
    def __init__(self, metric: Metric, split_data: Dict[str, SplitView], 
                 dataset_to_bundle: Dict[int, str]):
        self.metric = metric
        self.split_data = split_data
        self.dataset_to_bundle = dataset_to_bundle  # dataset id -> "train"/"val"/"test"
    
    def __call__(self, y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
        bundle_name = self.dataset_to_bundle.get(id(dataset))
        if bundle_name is None:
            raise ValueError("Unknown dataset")
        
        view = self.split_data.get(bundle_name)
        score = self.metric.compute(y_pred, view)
        return self.metric.name, score, self.metric.higher_is_better


class FevalAdapterFactory:
    """Feval适配器工厂"""
    
    @staticmethod
    def create(metric_names: List[str], split_data: Dict[str, SplitView],
               datasets: Dict[str, lgb.Dataset]) -> List[Callable]:
        dataset_to_bundle = {id(ds): name for name, ds in datasets.items()}
        adapters = []
        for name in metric_names:
            metric = MetricRegistry.get(name)
            adapters.append(FevalAdapter(metric, split_data, dataset_to_bundle))
        return adapters
