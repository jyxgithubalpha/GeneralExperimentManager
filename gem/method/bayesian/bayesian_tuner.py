"""
BayesianTuner - 贝叶斯模型 Optuna 参数调优器
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ..training_dataclasses import TrainConfig

from ..base import BaseTrainer, BaseTuner
from .bayesian_param_space import BayesianParamSpace


class BayesianTuner(BaseTuner):
    """
    贝叶斯模型 Optuna 参数调优器
    
    支持:
    - 串行/并行 trials
    - ask/tell 接口 (与 Ray 结合)
    - 基于历史的搜索空间收缩 (v1)
    - tuning_state 状态跟踪
    """
    
    def __init__(
        self,
        param_space: BayesianParamSpace,
        base_params: Dict[str, Any],
        n_trials: int = 50,
        timeout: Optional[int] = None,
        target_metric: str = "pearsonr_ic",
        seed: int = 42,
        direction: str = "maximize",
        use_shrinkage: bool = False,
        shrink_ratio: float = 0.5,
    ):
        self.param_space = param_space
        self.base_params = base_params
        self.n_trials = n_trials
        self.timeout = timeout
        self.target_metric = target_metric
        self.seed = seed
        self.direction = direction
        self.use_shrinkage = use_shrinkage
        self.shrink_ratio = shrink_ratio
        
        # 记录最近的调参结果
        self._last_best_params: Optional[Dict[str, Any]] = None
        self._last_best_value: Optional[float] = None
    
    def tune(
        self,
        views: "ProcessedViews",
        trainer: BaseTrainer,
        config: "TrainConfig",
        tuning_state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        调参 (串行)
        
        Args:
            views: 处理后的视图
            trainer: 训练器
            config: 训练配置
            tuning_state: 调参状态 (可选，用于搜索空间收缩)
            
        Returns:
            {"best_params": ..., "best_value": ..., "n_trials": ...}
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna is required for BayesianTuner")
        
        from ..training_dataclasses import TrainConfig as TC
        
        # 计算收缩后的搜索空间
        shrunk_space = None
        if self.use_shrinkage and tuning_state is not None:
            base_space = self.param_space.to_dict()
            shrunk_space = tuning_state.get_shrunk_space(base_space)
        
        def objective(trial) -> float:
            # 采样参数 (使用收缩后的空间)
            sampled_params = self.param_space.sample(trial, shrunk_space)
            params = {**self.base_params, **sampled_params}
            
            # 创建轻量配置
            tune_config = TC(
                params=params,
                num_boost_round=config.num_boost_round,
                early_stopping_rounds=config.early_stopping_rounds,
                feval_names=config.feval_names[:1],
                objective_name=config.objective_name,
                seed=self.seed,
                verbose_eval=0,
            )
            
            # 训练
            fit_result = trainer.fit(views, tune_config, mode="tune")
            
            # 获取验证集 MSE (取负值，因为我们要最小化 MSE 但 direction 可能是 maximize)
            if "val" in fit_result.evals_result and "mse" in fit_result.evals_result["val"]:
                mse = fit_result.evals_result["val"]["mse"][-1]
                return -mse  # 返回负 MSE，因为 direction 是 maximize
            return float('-inf')
        
        # 创建 study
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        
        # 如果有历史最佳参数，作为初始点
        if tuning_state is not None and tuning_state.last_best_params is not None:
            try:
                study.enqueue_trial(tuning_state.last_best_params)
            except:
                pass
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        best_params = {**self.base_params, **study.best_trial.params}
        
        # 保存结果
        self._last_best_params = best_params
        self._last_best_value = study.best_value
        
        return {
            "best_params": best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "shrunk_space_used": shrunk_space is not None,
        }
    
    def create_study(self) -> Any:
        """创建 Optuna study (用于 ask/tell 接口)"""
        try:
            import optuna
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            return optuna.create_study(direction=self.direction, sampler=sampler)
        except ImportError:
            raise ImportError("optuna is required for BayesianTuner")
    
    def ask(self, study, shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Ask for next trial parameters"""
        trial = study.ask()
        sampled_params = self.param_space.sample(trial, shrunk_space)
        params = {**self.base_params, **sampled_params}
        return trial, params
    
    def tell(self, study, trial, value: float) -> None:
        """Tell the study about a trial result"""
        study.tell(trial, value)
    
    @property
    def last_best_params(self) -> Optional[Dict[str, Any]]:
        return self._last_best_params
    
    @property
    def last_best_value(self) -> Optional[float]:
        return self._last_best_value
