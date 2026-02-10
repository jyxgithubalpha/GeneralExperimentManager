"""
LightGBMTrainer - LightGBM 模型训练器

支持:
- 本地 LightGBM 训练
- Ray Trainer 分布式训练
- 自定义目标函数和评估指标
- RayDataViews 输入
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ...experiment.experiment_dataclasses import RollingState

from ..base import BaseTrainer
from ..method_dataclasses import TrainConfig, FitResult, RayDataViews
from .lgb_adapter import LightGBMAdapter


class LightGBMTrainer(BaseTrainer):
    """
    LightGBM 模型训练器
    
    支持:
    - 本地 LightGBM 训练
    - Ray Trainer 分布式训练 (use_ray_trainer=True)
    - 自定义目标函数
    - 自定义评估指标
    - 样本权重
    """
    
    def __init__(
        self,
        adapter: Optional[LightGBMAdapter] = None,
        use_ray_trainer: bool = False,
        ray_trainer_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            adapter: LightGBM 数据适配器
            use_ray_trainer: 是否使用 Ray Trainer
            ray_trainer_config: Ray Trainer 配置
        """
        self.adapter = adapter or LightGBMAdapter()
        self.use_ray_trainer = use_ray_trainer
        self.ray_trainer_config = ray_trainer_config or {}
    
    def fit(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        """
        训练 LightGBM 模型
        
        Args:
            views: 处理后的视图
            config: 训练配置
            mode: "full" 或 "tune"
            sample_weights: 可选的样本权重 {"train": ..., "val": ...}
            
        Returns:
            FitResult 实例
        """
        if config.use_ray_trainer or self.use_ray_trainer:
            return self._fit_with_ray(views, config, mode, sample_weights)
        else:
            return self._fit_local(views, config, mode, sample_weights)
    
    def _fit_local(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        """本地 LightGBM 训练"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMTrainer")
        
        from ...utils.feval import FevalAdapterFactory
        from ...utils.objectives import ObjectiveFactory
        
        start_time = time.time()
        weights = sample_weights or {}
        
        # 构建 datasets
        dtrain = self.adapter.to_dataset(views.train, weight=weights.get("train"))
        dval = self.adapter.to_dataset(views.val, reference=dtrain, weight=weights.get("val"))
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
    
    def _fit_with_ray(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        """使用 Ray Trainer 分布式训练"""
        try:
            import ray
            from ray.train.lightgbm import LightGBMTrainer as RayLGBMTrainer
            from ray.train import ScalingConfig, RunConfig
        except ImportError:
            raise ImportError("ray[train] is required. Install with: pip install 'ray[train]'")
        
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMTrainer")
        
        from ..base.base_adapter import RayDataAdapter
        
        start_time = time.time()
        weights = sample_weights or {}
        
        # 转换为 Ray Data
        ray_views = RayDataAdapter.views_to_ray_views(views, weights)
        train_dataset = ray_views.train.to_ray_dataset()
        val_dataset = ray_views.val.to_ray_dataset()
        
        # 构建参数
        params = dict(config.params)
        params["seed"] = config.seed
        params["num_boost_round"] = config.num_boost_round
        
        # Ray Trainer 配置
        scaling_config = ScalingConfig(
            num_workers=self.ray_trainer_config.get("num_workers", 1),
            use_gpu=self.ray_trainer_config.get("use_gpu", False),
        )
        
        run_config = RunConfig(
            name="lgb_train",
            verbose=0 if mode == "tune" else 1,
        )
        
        # 创建 Ray LightGBM Trainer
        ray_trainer = RayLGBMTrainer(
            params=params,
            label_column="y",
            datasets={"train": train_dataset, "valid": val_dataset},
            scaling_config=scaling_config,
            run_config=run_config,
        )
        
        # 训练
        result = ray_trainer.fit()
        
        # 从 checkpoint 加载模型
        checkpoint = result.checkpoint
        model = None
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                model_path = Path(checkpoint_dir) / "model.txt"
                if model_path.exists():
                    model = lgb.Booster(model_file=str(model_path))
        
        train_time = time.time() - start_time
        
        # 构建 evals_result (从 metrics 中提取)
        evals_result = {"train": {}, "val": {}}
        if result.metrics:
            for key, value in result.metrics.items():
                if "train" in key.lower():
                    metric_name = key.replace("train_", "")
                    evals_result["train"][metric_name] = [value]
                elif "valid" in key.lower() or "val" in key.lower():
                    metric_name = key.replace("valid_", "").replace("val_", "")
                    evals_result["val"][metric_name] = [value]
        
        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=result.metrics.get("training_iteration", 1),
            params=params,
            seed=config.seed,
            train_time=train_time,
            checkpoint_path=checkpoint.path if checkpoint else None,
        )
    
    def fit_from_ray_views(
        self,
        ray_views: RayDataViews,
        config: TrainConfig,
        mode: str = "full",
    ) -> FitResult:
        """
        从 RayDataViews 训练
        
        Args:
            ray_views: Ray Data 视图
            config: 训练配置
            mode: "full" 或 "tune"
            
        Returns:
            FitResult 实例
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMTrainer")
        
        start_time = time.time()
        
        # 从 RayDataViews 构建 datasets
        dtrain = self.adapter.from_ray_bundle(ray_views.train)
        dval = self.adapter.from_ray_bundle(ray_views.val, reference=dtrain)
        
        # 构建参数
        params = dict(config.params)
        params["seed"] = config.seed
        
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
