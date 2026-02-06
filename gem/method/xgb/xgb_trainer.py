"""
XGBoostTrainer - XGBoost 模型训练器
"""
from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ..training_dataclasses import FitResult, TrainConfig

from ..base import BaseTrainer
from ...data.data_adapter import XGBoostAdapter


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost 模型训练器
    
    支持:
    - XGBoost
    - 自定义目标函数
    - 自定义评估指标
    """
    
    def __init__(
        self,
        adapter: Optional[XGBoostAdapter] = None,
    ):
        self.adapter = adapter or XGBoostAdapter()
    
    def fit(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        mode: str = "full",
    ) -> "FitResult":
        """
        训练 XGBoost 模型
        
        Args:
            views: 处理后的视图
            config: 训练配置
            mode: "full" 或 "tune"
            
        Returns:
            FitResult 实例
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for XGBoostTrainer")
        
        from ..training_dataclasses import FitResult
        
        start_time = time.time()
        
        # 构建 DMatrix
        dtrain = self.adapter.to_dataset(views.train)
        dval = self.adapter.to_dataset(views.val)
        
        # 构建参数
        params = dict(config.params)
        params["seed"] = config.seed
        
        # 设置 objective
        if config.objective_name == "regression":
            params["objective"] = "reg:squarederror"
        elif config.objective_name:
            params["objective"] = config.objective_name
        
        # 构建 evals
        evals = [(dtrain, "train"), (dval, "val")]
        
        # 构建 callbacks
        verbose = mode == "full"
        callbacks = []
        if not verbose:
            callbacks.append(xgb.callback.EarlyStopping(
                rounds=config.early_stopping_rounds,
                save_best=True,
            ))
        
        evals_result = {}
        
        # 训练
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=config.num_boost_round,
            evals=evals,
            early_stopping_rounds=config.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=config.verbose_eval if verbose else False,
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
