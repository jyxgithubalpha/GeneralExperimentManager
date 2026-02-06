"""
LightGBMTrainer - LightGBM 模型训练器
"""
from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ..training_dataclasses import FitResult, TrainConfig

from ..base import BaseTrainer
from ...data.data_adapter import LightGBMAdapter
from ...utils.feval import FevalAdapterFactory
from ...utils.objectives import ObjectiveFactory


class LightGBMTrainer(BaseTrainer):
    """
    LightGBM 模型训练器
    
    支持:
    - LightGBM
    - 自定义目标函数
    - 自定义评估指标
    """
    
    def __init__(
        self,
        adapter: Optional[LightGBMAdapter] = None,
    ):
        self.adapter = adapter or LightGBMAdapter()
    
    def fit(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        mode: str = "full",
    ) -> "FitResult":
        """
        训练 LightGBM 模型
        
        Args:
            views: 处理后的视图
            config: 训练配置
            mode: "full" 或 "tune"
            
        Returns:
            FitResult 实例
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMTrainer")
        
        from ..training_dataclasses import FitResult
        
        start_time = time.time()
        
        # 构建 datasets
        dtrain = self.adapter.to_dataset(views.train)
        dval = self.adapter.to_dataset(views.val, reference=dtrain)
        datasets = {"train": dtrain, "val": dval}
        
        # 构建参数
        params = dict(config.params)
        params["seed"] = config.seed
        
        # 获取 objective
        objective = ObjectiveFactory.get(
            config.objective_name,
            views=views,
            datasets=datasets,
        )
        if callable(objective):
            params["objective"] = objective
        else:
            params["objective"] = objective
        
        # 构建 feval
        split_data = {
            "train": views.train,
            "val": views.val, 
            "test": views.test
        }
        feval_list = FevalAdapterFactory.create(config.feval_names, split_data, datasets)
        
        # 构建 callbacks
        verbose = mode == "full"
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=config.early_stopping_rounds,
                first_metric_only=True,
                verbose=verbose,
            ),
            lgb.log_evaluation(period=config.verbose_eval if verbose else 0),
        ]
        
        evals_result = {}
        callbacks.append(lgb.record_evaluation(evals_result))
        
        # 训练
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=config.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            feval=feval_list if feval_list else None,
            callbacks=callbacks,
        )
        
        train_time = time.time() - start_time
        
        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=model.best_iteration,
            params=params,
            seed=config.seed,
            train_time=train_time,
        )
