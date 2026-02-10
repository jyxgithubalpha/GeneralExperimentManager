"""
Method - 统一训练接口

包含:
- BaseMethod: 统一训练接口，组合 Transform, Adapter, Tuner, Trainer, Evaluator, ImportanceExtractor

流程:
1. 从 SplitViews 利用 train/val 计算 X/y 的阈值, 均值, std 等
2. 对 train/val/test 做 transform 变换 (包括利用 RollingState 进行因子加权)
3. 在 adapter 里把 pl.DataFrame 转化成 numpy 为基础的 ray data
4. 得到 numpy 数据然后变换成 lgb.dataset
5. 使用 RollingState 中前面已有的超参数作为起点进行搜索训练
6. 使用 ray tune 和 optuna 结合搜索
7. 使用 ray trainer 和 lgb trainer 结合来训练
8. 评估并且根据结果更新状态

输出 MethodOutput:
- best_params
- tune_result
- fit_result
- metrics_eval
- importance_vector
- transform_stats
- state_update (用于更新 RollingState)
- model_artifacts
"""

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...experiment.experiment_dataclasses import RollingState

from ...data.data_dataclasses import ProcessedViews, SplitViews
from ..method_dataclasses import (
    MethodOutput, TrainConfig, TuneConfig, TuneResult, 
    FitResult, TransformStats, RayDataViews
)

from .base_trainer import BaseTrainer
from .base_evaluator import BaseEvaluator
from .base_importance_extractor import BaseImportanceExtractor
from .base_tuner import BaseTuner
from .base_transform import BaseTransformPipeline
from .base_adapter import BaseAdapter


class BaseMethod:
    """
    Method - 统一训练接口
    
    内部包含:
    - BaseTransformPipeline: 数据变换 pipeline
    - BaseAdapter: 数据适配器
    - BaseTuner: 超参调优器 (Optuna/Ray Tune)
    - BaseTrainer: 训练器 (LightGBM/Ray Trainer)
    - BaseEvaluator: 评估器
    - BaseImportanceExtractor: 特征重要性提取器
    
    输出 MethodOutput
    """
    
    def __init__(
        self,
        framework: str = "lightgbm",
        transform_pipeline: Optional[BaseTransformPipeline] = None,
        adapter: Optional[BaseAdapter] = None,
        trainer: Optional[BaseTrainer] = None,
        evaluator: Optional[BaseEvaluator] = None,
        importance_extractor: Optional[BaseImportanceExtractor] = None,
        tuner: Optional[BaseTuner] = None,
        **component_kwargs
    ):
        """
        初始化 Method
        
        Args:
            framework: 机器学习框架 ("lightgbm")
            transform_pipeline: 数据变换 pipeline (可选)
            adapter: 数据适配器 (可选)
            trainer: 自定义训练器 (可选)
            evaluator: 自定义评估器 (可选)
            importance_extractor: 自定义特征重要性提取器 (可选)
            tuner: 自定义调优器 (可选)
            component_kwargs: 组件初始化参数
        """
        self.framework = framework
        self.transform_pipeline = transform_pipeline
        self.adapter = adapter
        self.tuner = tuner
        self.trainer = trainer
        self.evaluator = evaluator
        self.importance_extractor = importance_extractor
    
    def run(
        self,
        views: ProcessedViews,
        config: TrainConfig,
        do_tune: bool = True,
        save_dir: Optional[Path] = None,
        rolling_state: Optional["RollingState"] = None,
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> MethodOutput:
        """
        运行完整流程: tune -> train -> eval -> importance
        
        Args:
            views: 处理后的视图
            config: 训练配置
            do_tune: 是否进行参数调优
            save_dir: 保存目录
            rolling_state: 滚动状态 (用于热启动和状态更新)
            sample_weights: 可选的样本权重 {"train": ..., "val": ...}
            
        Returns:
            MethodOutput 实例
        """
        # 1. Tune (可选)
        best_params = config.params.copy()
        tune_result = None
        
        if do_tune and self.tuner is not None:
            tune_result = self.tuner.tune(
                views, 
                self.trainer, 
                config,
                rolling_state=rolling_state,
            )
            best_params = tune_result.best_params
        
        # 2. Train with best params
        final_config = TrainConfig(
            params=best_params,
            num_boost_round=config.num_boost_round,
            early_stopping_rounds=config.early_stopping_rounds,
            feval_names=config.feval_names,
            objective_name=config.objective_name,
            seed=config.seed,
            verbose_eval=config.verbose_eval,
            use_ray_trainer=config.use_ray_trainer,
        )
        
        fit_result = self.trainer.fit(
            views, 
            final_config, 
            mode="full",
            sample_weights=sample_weights,
        )
        
        # 3. Evaluate
        metrics_eval = self.evaluator.evaluate(fit_result.model, views)
        
        # 4. Extract importance
        importance_vector, importance_df = self.importance_extractor.extract(
            fit_result.model,
            views.train.feature_names,
        )
        fit_result.feature_importance = importance_df
        
        # 5. Save artifacts
        model_artifacts = {}
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = save_dir / "model.txt"
            if hasattr(fit_result.model, 'save_model'):
                fit_result.model.save_model(str(model_path))
                model_artifacts["model"] = model_path
            
            # Save importance
            if importance_df is not None:
                importance_path = save_dir / "feature_importance.csv"
                importance_df.write_csv(str(importance_path))
                model_artifacts["importance"] = importance_path
        
        # 6. Get feature names hash
        feature_names_hash = views.train.get_feature_names_hash()
        
        # 7. 构建状态更新增量
        state_update = {
            "importance_vector": importance_vector,
            "feature_names_hash": feature_names_hash,
            "best_params": best_params,
        }
        if tune_result is not None:
            state_update["best_objective"] = tune_result.best_value
        
        return MethodOutput(
            best_params=best_params,
            metrics_eval=metrics_eval,
            importance_vector=importance_vector,
            feature_names_hash=feature_names_hash,
            tune_result=tune_result,
            fit_result=fit_result,
            state_update=state_update,
            model_artifacts=model_artifacts,
        )
    
    def run_full_pipeline(
        self,
        split_views: SplitViews,
        config: TrainConfig,
        tune_config: Optional[TuneConfig] = None,
        rolling_state: Optional["RollingState"] = None,
        save_dir: Optional[Path] = None,
    ) -> MethodOutput:
        """
        运行完整 pipeline: transform -> adapt -> tune -> train -> eval -> update state
        
        Args:
            split_views: 原始 split 视图
            config: 训练配置
            tune_config: 搜索配置 (可选)
            rolling_state: 滚动状态
            save_dir: 保存目录
            
        Returns:
            MethodOutput 实例
        """
        # 1. Transform
        transform_stats = None
        if self.transform_pipeline is not None:
            processed_views, transform_stats = self.transform_pipeline.fit_transform(
                split_views,
                rolling_state=rolling_state,
            )
        else:
            # 无 transform 时直接使用
            processed_views = ProcessedViews(
                train=split_views.train,
                val=split_views.val,
                test=split_views.test,
                split_spec=split_views.split_spec,
            )
        
        # 2. 获取样本权重 (如果有)
        sample_weights = None
        if rolling_state is not None:
            from ...experiment.experiment_dataclasses import SampleWeightState
            weight_state = rolling_state.get_state(SampleWeightState)
            if weight_state is not None and hasattr(weight_state, 'get_weights_for_view'):
                sample_weights = {
                    "train": weight_state.get_weights_for_view(processed_views.train),
                    "val": weight_state.get_weights_for_view(processed_views.val),
                }
        
        # 3. 运行主流程
        do_tune = tune_config is not None and tune_config.n_trials > 0
        output = self.run(
            views=processed_views,
            config=config,
            do_tune=do_tune,
            save_dir=save_dir,
            rolling_state=rolling_state,
            sample_weights=sample_weights,
        )
        
        # 4. 附加 transform_stats
        output.transform_stats = transform_stats
        
        return output
    
    def update_rolling_state(
        self,
        output: MethodOutput,
        rolling_state: "RollingState",
        feature_names: Optional[list] = None,
        alpha: float = 0.3,
    ) -> "RollingState":
        """
        根据 MethodOutput 更新 RollingState
        
        Args:
            output: Method 输出
            rolling_state: 当前滚动状态
            feature_names: 特征名列表
            alpha: EMA 平滑系数
            
        Returns:
            更新后的 RollingState
        """
        # 更新特征重要性
        rolling_state.update_importance(
            output.importance_vector,
            feature_names=feature_names,
            alpha=alpha,
        )
        
        # 更新调参状态
        if output.tune_result is not None:
            rolling_state.update_tuning(
                output.best_params,
                output.tune_result.best_value,
            )
        
        return rolling_state
