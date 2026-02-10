"""
Hook系统 - 扩展能力
"""



from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import polars as pl

from ..method.method_dataclasses import EvalResult, FitResult
from ..data.data_dataclasses import SplitSpec
from ..data.data_dataclasses import SplitData

if TYPE_CHECKING:
    from ..experiment.store import Store


class HookEvent(Enum):
    """Hook事件类型"""
    BEFORE_SPLIT = "before_split"
    ON_SPLIT_END = "on_split_end"
    AFTER_TRANSFORM = "after_transform"
    BEFORE_TRAIN = "before_train"
    AFTER_TRAIN = "after_train"
    BEFORE_EVAL = "before_eval"
    AFTER_EVAL = "after_eval"
    ON_TRIAL_END = "on_trial_end"
    ON_EXPERIMENT_END = "on_experiment_end"


@dataclass
class HookContext:
    """Hook上下文"""
    event: HookEvent
    split_spec: Optional[SplitSpec] = None
    split_data: Optional[SplitData] = None
    fit_result: Optional[FitResult] = None
    eval_result: Optional[EvalResult] = None
    params: Optional[Dict[str, Any]] = None
    store: Optional["Store"] = None
    mode: str = "full"  # "tune" or "full"


class Hook(ABC):
    """Hook基类"""
    name: str = "base_hook"
    events: List[HookEvent] = []
    
    @abstractmethod
    def __call__(self, ctx: HookContext) -> None:
        pass


class SaveModelHook(Hook):
    """保存模型Hook"""
    name = "save_model"
    events = [HookEvent.AFTER_TRAIN]
    
    def __call__(self, ctx: HookContext) -> None:
        if ctx.mode == "tune" or ctx.fit_result is None or ctx.store is None:
            return
        path = ctx.store.get_model_path(ctx.split_spec.split_id)
        ctx.fit_result.model.save_model(str(path))


class FeatureImportanceHook(Hook):
    """特征重要性Hook"""
    name = "feature_importance"
    events = [HookEvent.AFTER_TRAIN]
    
    def __init__(self, importance_type: str = "gain"):
        self.importance_type = importance_type
    
    def __call__(self, ctx: HookContext) -> None:
        if ctx.mode == "tune" or ctx.fit_result is None or ctx.store is None:
            return
        
        model = ctx.fit_result.model
        importance = model.feature_importance(importance_type=self.importance_type)
        feature_names = model.feature_name()
        
        df = pl.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort("importance", descending=True)
        
        ctx.fit_result.feature_importance = df
        path = ctx.store.get_artifact_path(ctx.split_spec.split_id, "feature_importance.csv")
        df.write_csv(path)


class SavePredictionsHook(Hook):
    """保存预测结果Hook"""
    name = "save_predictions"
    events = [HookEvent.AFTER_EVAL]
    
    def __call__(self, ctx: HookContext) -> None:
        if ctx.mode == "tune" or ctx.eval_result is None or ctx.store is None:
            return
        
        if ctx.eval_result.predictions is not None:
            path = ctx.store.get_artifact_path(ctx.split_spec.split_id, "predictions.npy")
            np.save(path, ctx.eval_result.predictions)


class TrainLogHook(Hook):
    """训练日志Hook"""
    name = "train_log"
    events = [HookEvent.AFTER_TRAIN]
    
    def __call__(self, ctx: HookContext) -> None:
        if ctx.fit_result is None:
            return
        
        print(f"[Split {ctx.split_spec.split_id}] "
              f"Best iteration: {ctx.fit_result.best_iteration}, "
              f"Train time: {ctx.fit_result.train_time:.2f}s")


class HookManager:
    """Hook管理器"""
    
    def __init__(self):
        self._hooks: Dict[HookEvent, List[Hook]] = {e: [] for e in HookEvent}
    
    def register(self, hook: Hook) -> "HookManager":
        for event in hook.events:
            self._hooks[event].append(hook)
        return self
    
    def trigger(self, ctx: HookContext) -> None:
        for hook in self._hooks[ctx.event]:
            try:
                hook(ctx)
            except Exception as e:
                print(f"Hook '{hook.name}' failed: {e}")
    
    def clear(self) -> None:
        for event in HookEvent:
            self._hooks[event] = []
