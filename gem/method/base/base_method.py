"""
Method - 统一训练接口

包含:
- Method: 统一训练接口，组合 Tuner, Trainer, Evaluator, ImportanceExtractor

输出 MethodOutput:
- best_params
- metrics_search (可选)
- metrics_eval
- importance_vector: np.ndarray (与当前 feature_names 对齐)
- aux_state_delta (可选)
- model_artifacts
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ..training_dataclasses import MethodOutput, TrainConfig

from .base_trainer import BaseTrainer
from .base_evaluator import BaseEvaluator
from .base_importance_extractor import BaseImportanceExtractor
from .base_tuner import BaseTuner


class Method:
    """
    Method - 统一训练接口
    
    内部包含:
    - BaseTuner (Optuna)
    - BaseTrainer
    - BaseEvaluator
    - BaseImportanceExtractor
    
    输出 MethodOutput
    """
    
    def __init__(
        self,
        framework: str = "lightgbm",
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
            trainer: 自定义训练器 (可选)
            evaluator: 自定义评估器 (可选)
            importance_extractor: 自定义特征重要性提取器 (可选)
            tuner: 自定义调优器 (可选)
            component_kwargs: 组件初始化参数
        """
        self.tuner = tuner
        self.trainer = trainer
        self.evaluator = evaluator
        self.importance_extractor = importance_extractor
    
    def run(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        do_tune: bool = True,
        save_dir: Optional[Path] = None,
    ) -> "MethodOutput":
        """
        运行完整流程: tune -> train -> eval -> importance
        
        Args:
            views: 处理后的视图
            config: 训练配置
            do_tune: 是否进行参数调优
            save_dir: 保存目录
            
        Returns:
            MethodOutput 实例
        """
        from ..training_dataclasses import MethodOutput, TrainConfig as TC
        
        # 1. Tune (可选)
        best_params = config.params.copy()
        metrics_search = None
        
        if do_tune and self.tuner is not None:
            tune_result = self.tuner.tune(views, self.trainer, config)
            best_params = tune_result["best_params"]
            metrics_search = {"best_value": tune_result["best_value"]}
        
        # 2. Train with best params
        final_config = TC(
            params=best_params,
            num_boost_round=config.num_boost_round,
            early_stopping_rounds=config.early_stopping_rounds,
            feval_names=config.feval_names,
            objective_name=config.objective_name,
            seed=config.seed,
            verbose_eval=config.verbose_eval,
        )
        
        fit_result = self.trainer.fit(views, final_config, mode="full")
        
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
            fit_result.model.save_model(str(model_path))
            model_artifacts["model"] = model_path
            
            # Save importance
            importance_path = save_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            model_artifacts["importance"] = importance_path
        
        # 6. Get feature names hash
        feature_names_hash = views.train.get_feature_names_hash()
        
        return MethodOutput(
            best_params=best_params,
            metrics_eval=metrics_eval,
            importance_vector=importance_vector,
            feature_names_hash=feature_names_hash,
            metrics_search=metrics_search,
            model_artifacts=model_artifacts,
            fit_result=fit_result,
        )
