"""
CatBoostTrainer - CatBoost 模型训练器
"""
from __future__ import annotations

import time
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ..training_dataclasses import FitResult, TrainConfig

from ..base import BaseTrainer
from ...data.data_adapter import CatBoostAdapter


class CatBoostTrainer(BaseTrainer):
    """
    CatBoost 模型训练器
    
    支持:
    - CatBoost
    - 自动类别特征处理
    - 自定义评估指标
    """
    
    def __init__(
        self,
        adapter: Optional[CatBoostAdapter] = None,
        cat_features: Optional[List[int]] = None,
    ):
        self.adapter = adapter or CatBoostAdapter(cat_features=cat_features)
        self.cat_features = cat_features
    
    def fit(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        mode: str = "full",
    ) -> "FitResult":
        """
        训练 CatBoost 模型
        
        Args:
            views: 处理后的视图
            config: 训练配置
            mode: "full" 或 "tune"
            
        Returns:
            FitResult 实例
        """
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError:
            raise ImportError("catboost is required for CatBoostTrainer")
        
        from ..training_dataclasses import FitResult
        
        start_time = time.time()
        
        # 构建 Pool
        train_pool = self.adapter.to_dataset(views.train)
        val_pool = self.adapter.to_dataset(views.val)
        
        # 构建参数
        params = dict(config.params)
        params["random_seed"] = config.seed
        
        # 设置 loss_function
        if config.objective_name == "regression":
            params["loss_function"] = "RMSE"
        elif config.objective_name:
            params["loss_function"] = config.objective_name
        
        # 设置迭代次数和早停
        params["iterations"] = config.num_boost_round
        params["early_stopping_rounds"] = config.early_stopping_rounds
        
        # 设置 verbose
        verbose = mode == "full"
        params["verbose"] = config.verbose_eval if verbose else False
        
        # 创建模型并训练
        model = CatBoostRegressor(**params)
        
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )
        
        train_time = time.time() - start_time
        
        # 构建 evals_result
        evals_result = {
            "train": {},
            "val": {},
        }
        
        # 获取训练历史
        if hasattr(model, "evals_result_"):
            evals = model.evals_result_
            if "learn" in evals:
                evals_result["train"] = evals["learn"]
            if "validation" in evals:
                evals_result["val"] = evals["validation"]
        
        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=model.get_best_iteration() if hasattr(model, "get_best_iteration") else None,
            params=params,
            seed=config.seed,
            train_time=train_time,
        )
