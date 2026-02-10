"""
LightGBMTuner - LightGBM 超参调优器

支持:
- 纯 Optuna 串行搜索
- Ray Tune + Optuna 并行搜索
- RollingState 历史超参热启动
- 搜索空间收缩
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ...experiment.experiment_dataclasses import TuningState, RollingState

from ..base import BaseTrainer, BaseTuner
from ..method_dataclasses import TrainConfig, TuneConfig, TuneResult
from .lgb_param_space import LightGBMParamSpace


class LightGBMTuner(BaseTuner):
    """
    LightGBM 超参调优器
    
    支持:
    - 纯 Optuna 串行搜索
    - Ray Tune + Optuna 并行搜索  
    - RollingState 历史超参作为热启动点
    - 基于历史的搜索空间收缩
    """
    
    def __init__(
        self,
        param_space: Optional[LightGBMParamSpace] = None,
        base_params: Optional[Dict[str, Any]] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        target_metric: str = "pearsonr_ic",
        seed: int = 42,
        direction: str = "maximize",
        use_shrinkage: bool = True,
        shrink_ratio: float = 0.5,
        use_ray_tune: bool = False,
        parallel_trials: int = 1,
        use_warm_start: bool = True,
    ):
        """
        Args:
            param_space: 参数空间 (None 时使用默认)
            base_params: 基础参数
            n_trials: 搜索次数
            timeout: 超时时间 (秒)
            target_metric: 目标指标
            seed: 随机种子
            direction: 优化方向 (maximize/minimize)
            use_shrinkage: 是否使用搜索空间收缩
            shrink_ratio: 收缩比例
            use_ray_tune: 是否使用 Ray Tune
            parallel_trials: 并行 trial 数
            use_warm_start: 是否使用历史超参热启动
        """
        self.param_space = param_space or LightGBMParamSpace()
        self.base_params = base_params or {}
        self.n_trials = n_trials
        self.timeout = timeout
        self.target_metric = target_metric
        self.seed = seed
        self.direction = direction
        self.use_shrinkage = use_shrinkage
        self.shrink_ratio = shrink_ratio
        self.use_ray_tune = use_ray_tune
        self.parallel_trials = parallel_trials
        self.use_warm_start = use_warm_start
        
        # 记录最近的调参结果
        self._last_best_params: Optional[Dict[str, Any]] = None
        self._last_best_value: Optional[float] = None
    
    def tune(
        self,
        views: "ProcessedViews",
        trainer: BaseTrainer,
        config: TrainConfig,
        tuning_state: Optional["TuningState"] = None,
        rolling_state: Optional["RollingState"] = None,
    ) -> TuneResult:
        """
        执行超参搜索
        
        Args:
            views: 处理后的视图
            trainer: 训练器
            config: 训练配置
            tuning_state: 调参状态 (用于搜索空间收缩和热启动)
            rolling_state: 滚动状态 (备选热启动来源)
            
        Returns:
            TuneResult
        """
        # 从 rolling_state 获取 tuning_state
        if tuning_state is None and rolling_state is not None:
            from ...experiment.experiment_dataclasses import TuningState as TS
            tuning_state = rolling_state.get_state(TS)
        
        if self.use_ray_tune and self.parallel_trials > 1:
            return self._tune_with_ray(views, trainer, config, tuning_state)
        else:
            return self._tune_with_optuna(views, trainer, config, tuning_state)
    
    def _tune_with_optuna(
        self,
        views: "ProcessedViews",
        trainer: BaseTrainer,
        config: TrainConfig,
        tuning_state: Optional["TuningState"] = None,
    ) -> TuneResult:
        """使用 Optuna 串行搜索"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna is required for LightGBMTuner")
        
        start_time = time.time()
        
        # 计算收缩后的搜索空间
        shrunk_space = None
        warm_start_used = False
        
        if self.use_shrinkage and tuning_state is not None:
            base_space = self.param_space.to_dict()
            shrunk_space = tuning_state.get_shrunk_space(base_space)
        
        # 创建 objective
        def objective(trial) -> float:
            sampled_params = self.param_space.sample(trial, shrunk_space)
            params = {**self.base_params, **sampled_params}
            
            # 创建轻量配置
            tune_config = config.for_tuning(params, self.seed)
            
            # 训练
            fit_result = trainer.fit(views, tune_config, mode="tune")
            
            # 获取验证集指标
            metric_name = tune_config.feval_names[0]
            if metric_name in fit_result.evals_result.get("val", {}):
                scores = fit_result.evals_result["val"][metric_name]
                return scores[fit_result.best_iteration - 1]
            return 0.0
        
        # 创建 study
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        
        # 热启动: 将历史最佳参数作为初始 trial
        if self.use_warm_start and tuning_state is not None:
            if tuning_state.last_best_params is not None:
                try:
                    # 过滤出参数空间内的参数
                    warm_params = {
                        k: v for k, v in tuning_state.last_best_params.items()
                        if k in self.param_space.get_param_names()
                    }
                    if warm_params:
                        study.enqueue_trial(warm_params)
                        warm_start_used = True
                except Exception:
                    pass
        
        # 执行搜索
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        best_params = {**self.base_params, **study.best_trial.params}
        
        # 保存结果
        self._last_best_params = best_params
        self._last_best_value = study.best_value
        
        # 收集所有 trials
        all_trials = [
            {"params": t.params, "value": t.value, "state": str(t.state)}
            for t in study.trials
        ]
        
        return TuneResult(
            best_params=best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            all_trials=all_trials,
            search_time=time.time() - start_time,
            warm_start_used=warm_start_used,
            shrunk_space_used=shrunk_space is not None,
        )
    
    def _tune_with_ray(
        self,
        views: "ProcessedViews",
        trainer: BaseTrainer,
        config: TrainConfig,
        tuning_state: Optional["TuningState"] = None,
    ) -> TuneResult:
        """使用 Ray Tune + Optuna 并行搜索"""
        try:
            import ray
            from ray import tune
            from ray.tune.search.optuna import OptunaSearch
        except ImportError:
            raise ImportError("ray[tune] is required. Install with: pip install 'ray[tune]'")
        
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna is required for LightGBMTuner")
        
        start_time = time.time()
        
        # 计算收缩后的搜索空间
        shrunk_space = None
        warm_start_used = False
        
        if self.use_shrinkage and tuning_state is not None:
            base_space = self.param_space.to_dict()
            shrunk_space = tuning_state.get_shrunk_space(base_space)
        
        # 构建 Ray Tune 搜索空间
        ray_search_space = self.param_space.to_ray_tune_space(shrunk_space)
        
        # 热启动点
        initial_params = None
        if self.use_warm_start and tuning_state is not None:
            if tuning_state.last_best_params is not None:
                initial_params = [
                    {k: v for k, v in tuning_state.last_best_params.items()
                     if k in self.param_space.get_param_names()}
                ]
                if initial_params and initial_params[0]:
                    warm_start_used = True
                else:
                    initial_params = None
        
        # 创建 Optuna 搜索器
        optuna_search = OptunaSearch(
            metric=self.target_metric,
            mode="max" if self.direction == "maximize" else "min",
            seed=self.seed,
            points_to_evaluate=initial_params,
        )
        
        # 定义 trainable
        def trainable(ray_config):
            params = {**self.base_params, **ray_config}
            tune_config = config.for_tuning(params, self.seed)
            
            fit_result = trainer.fit(views, tune_config, mode="tune")
            
            metric_name = tune_config.feval_names[0]
            if metric_name in fit_result.evals_result.get("val", {}):
                scores = fit_result.evals_result["val"][metric_name]
                score = scores[fit_result.best_iteration - 1]
            else:
                score = 0.0
            
            return {self.target_metric: score}
        
        # 执行搜索
        analysis = tune.run(
            trainable,
            config=ray_search_space,
            num_samples=self.n_trials,
            search_alg=optuna_search,
            resources_per_trial={"cpu": 1, "gpu": 0},
            verbose=0,
        )
        
        # 获取最佳结果
        best_trial = analysis.get_best_trial(self.target_metric, "max" if self.direction == "maximize" else "min")
        best_params = {**self.base_params, **best_trial.config}
        best_value = best_trial.last_result[self.target_metric]
        
        # 保存结果
        self._last_best_params = best_params
        self._last_best_value = best_value
        
        # 收集所有 trials
        all_trials = [
            {"params": t.config, "value": t.last_result.get(self.target_metric), "state": str(t.status)}
            for t in analysis.trials
        ]
        
        return TuneResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(analysis.trials),
            all_trials=all_trials,
            search_time=time.time() - start_time,
            warm_start_used=warm_start_used,
            shrunk_space_used=shrunk_space is not None,
        )
    
    def create_study(self) -> Any:
        """创建 Optuna study (用于 ask/tell 接口)"""
        try:
            import optuna
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            return optuna.create_study(direction=self.direction, sampler=sampler)
        except ImportError:
            raise ImportError("optuna is required for LightGBMTuner")
    
    def ask(
        self, 
        study, 
        shrunk_space: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Ask for next trial parameters (ask/tell 接口)"""
        trial = study.ask()
        sampled_params = self.param_space.sample(trial, shrunk_space)
        params = {**self.base_params, **sampled_params}
        return trial, params
    
    def tell(self, study, trial, value: float) -> None:
        """Tell the study about a trial result (ask/tell 接口)"""
        study.tell(trial, value)
    
    @property
    def last_best_params(self) -> Optional[Dict[str, Any]]:
        return self._last_best_params
    
    @property
    def last_best_value(self) -> Optional[float]:
        return self._last_best_value
    
    @classmethod
    def from_tune_config(
        cls,
        tune_config: TuneConfig,
        base_params: Optional[Dict[str, Any]] = None,
        param_space: Optional[LightGBMParamSpace] = None,
    ) -> "LightGBMTuner":
        """从 TuneConfig 创建 Tuner"""
        return cls(
            param_space=param_space,
            base_params=base_params or {},
            n_trials=tune_config.n_trials,
            timeout=tune_config.timeout,
            target_metric=tune_config.target_metric,
            seed=tune_config.seed,
            direction=tune_config.direction,
            use_shrinkage=tune_config.shrink_ratio > 0,
            shrink_ratio=tune_config.shrink_ratio,
            use_ray_tune=tune_config.use_ray_tune,
            parallel_trials=tune_config.parallel_trials,
            use_warm_start=tune_config.use_warm_start,
        )
