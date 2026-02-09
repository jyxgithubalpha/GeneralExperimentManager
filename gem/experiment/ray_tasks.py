"""
Ray Tasks - Ray 任务定义

Ray 单元:
- CPU tasks: update_state, aggregate_importance
- GPU tasks: run_trial, run_final_train
- Split 入口 (CPU): run_split (内部提交 trial/final_train GPU tasks)

关键约束:
- run_split 不占 GPU; GPU 留给 trial/final_train tasks
- artifacts 在 exp/split_id/(trial_id)/... 落盘
- state 每步可恢复
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.data_dataclasses import (
    ProcessedViews,
    GlobalStore,
    SplitViews,
)
from .experiment_manager import SplitResult
from .experiment_manager import (
    ExperimentConfig,
    SplitTask,
)
from .state_dataclasses import RollingState
from ..method.training_dataclasses import TrainConfig


# =============================================================================
# 1. 检查 Ray 可用性
# =============================================================================

def _check_ray():
    """检查 Ray 是否可用"""
    try:
        import ray
        return ray
    except ImportError:
        raise ImportError("ray is required. Install with: pip install ray")


# =============================================================================
# 2. Ray Remote Functions (will be wrapped)
# =============================================================================

def _run_split_impl(
    task: SplitTask,
    global_store: GlobalStore,
    rolling_state: Optional[RollingState],
    config: ExperimentConfig,
    train_config: TrainConfig,
    method_config: Optional[Dict[str, Any]] = None,
    data_processor_config: Optional[Dict[str, Any]] = None,
) -> SplitResult:
    """
    执行单个 split 的实现
    
    流程:
    1. 索引取数 -> build_views
    2. DataProcessor -> fit_transform
    3. Adapter -> to_backend_dataset
    4. Method.run() -> tune/train/eval/importance
    5. 返回 SplitResult
    
    Args:
        method_config: 包含 trainer, evaluator, importance_extractor, tuner 的配置字典
        data_processor_config: DataProcessor Hydra 配置
    """
    from hydra.utils import instantiate
    from ..method.base import Method
    
    split_id = task.split_id
    splitspec = task.splitspec
    
    try:
        # 1. Build views
        idx_train = global_store.get_indices_by_dates(splitspec.train_date_list)
        idx_val = global_store.get_indices_by_dates(splitspec.val_date_list)
        idx_test = global_store.get_indices_by_dates(splitspec.test_date_list)
        
        # 检查是否有空集，如果有则跳过该 split
        if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
            empty_sets = []
            if len(idx_train) == 0:
                empty_sets.append("train")
            if len(idx_val) == 0:
                empty_sets.append("val")
            if len(idx_test) == 0:
                empty_sets.append("test")
            
            return SplitResult(
                split_id=split_id,
                metrics={},
                skipped=True,
                skip_reason=f"Empty sets: {', '.join(empty_sets)}",
            )
        
        split_views = SplitViews(
            train=global_store.take(idx_train),
            val=global_store.take(idx_val),
            test=global_store.take(idx_test),
            split_spec=splitspec,
        )
        
        # 2. DataProcessor - 从配置实例化
        processor = instantiate(data_processor_config)
        processed_views = processor.fit_transform(split_views, rolling_state)
        
        # 3. Setup Method - 从配置实例化组件
        trainer = None
        evaluator = None
        importance_extractor = None
        tuner = None
        
        # 使用 Hydra instantiate 加载组件
        trainer = instantiate(method_config["trainer"])
        evaluator = instantiate(method_config["evaluator"])
        importance_extractor = instantiate(method_config["importance_extractor"])
        if config.n_trials > 0:
            tuner_cfg = method_config["tuner"].copy() if hasattr(method_config["tuner"], 'copy') else dict(method_config["tuner"])
            tuner = instantiate(
                tuner_cfg,
                base_params=train_config.params,
                n_trials=config.n_trials,
                timeout=config.trial_timeout,
                seed=config.seed + split_id,
            )
        
        method = Method(
            trainer=trainer,
            evaluator=evaluator,
            importance_extractor=importance_extractor,
            tuner=tuner,
        )
        
        # 4. Run method
        save_dir = Path(config.output_dir) / f"split_{split_id}"
        method_output = method.run(
            views=processed_views,
            config=train_config,
            do_tune=(tuner is not None),
            save_dir=save_dir,
        )
        
        # 5. Build SplitResult
        # Flatten metrics
        metrics_flat = {}
        for mode, eval_result in method_output.metrics_eval.items():
            metrics_flat.update(eval_result.metrics)
        
        return SplitResult(
            split_id=split_id,
            importance_vector=method_output.importance_vector,
            feature_names_hash=method_output.feature_names_hash,
            metrics=metrics_flat,
        )
        
    except Exception as e:
        import traceback
        # 创建一个带有错误信息的 metrics 字典
        error_metrics = {"error": f"{str(e)}\n{traceback.format_exc()}"}
        return SplitResult(
            split_id=split_id,
            importance_vector=np.array([]),
            feature_names_hash="",
            metrics=error_metrics,
        )


def _run_trial_impl(
    trial_id: int,
    params: Dict[str, Any],
    processed_views: ProcessedViews,
    train_config: TrainConfig,
    seed: int,
) -> Tuple[int, float, Dict[str, Any]]:
    """
    执行单个 trial 的实现 (GPU task)
    
    Returns:
        (trial_id, score, params)
    """
    from ..method.trainer import Trainer
    
    try:
        # 创建 trial 配置
        trial_config = TrainConfig(
            params=params,
            num_boost_round=train_config.num_boost_round,
            early_stopping_rounds=train_config.early_stopping_rounds,
            feval_names=train_config.feval_names[:1],
            objective_name=train_config.objective_name,
            seed=seed,
            verbose_eval=0,
        )
        
        trainer = Trainer()
        fit_result = trainer.fit(processed_views, trial_config, mode="tune")
        
        # 获取验证集指标
        metric_name = trial_config.feval_names[0]
        if metric_name in fit_result.evals_result.get("val", {}):
            scores = fit_result.evals_result["val"][metric_name]
            score = scores[fit_result.best_iteration - 1]
        else:
            score = 0.0
        
        return trial_id, score, params
        
    except Exception as e:
        return trial_id, 0.0, params


def _run_final_train_impl(
    processed_views: ProcessedViews,
    train_config: TrainConfig,
    save_dir: Path,
) -> MethodOutput:
    """
    执行最终训练的实现 (GPU task)
    """
    from ..method.method import Method
    from ..method.trainer import Trainer
    from ..method.evaluator import Evaluator
    from ..method.importance_extractor import ImportanceExtractor
    
    method = Method(
        trainer=Trainer(),
        evaluator=Evaluator(),
        importance_extractor=ImportanceExtractor(),
        tuner=None,
    )
    
    return method.run(
        views=processed_views,
        config=train_config,
        do_tune=False,
        save_dir=save_dir,
    )


def _update_state_impl(
    prev_state: Optional[RollingState],
    split_result: SplitResult,
    config: StatePolicyConfig,
) -> RollingState:
    """更新状态的实现"""
    from .state_policy import update_state
    return update_state(prev_state, split_result, config)


def _aggregate_importance_impl(
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> np.ndarray:
    """聚合重要性的实现"""
    from .state_policy import aggregate_bucket_results
    return aggregate_bucket_results(results, config)


def _update_state_from_bucket_impl(
    prev_state: Optional[RollingState],
    results: List[SplitResult],
    config: StatePolicyConfig,
) -> RollingState:
    """从 bucket 更新状态的实现"""
    from .state_policy import update_state_from_bucket_results
    return update_state_from_bucket_results(prev_state, results, config)


# =============================================================================
# 3. Ray Remote Wrappers
# =============================================================================

class RayTaskManager:
    """
    Ray 任务管理器
    
    负责:
    - 创建 Ray remote functions
    - 管理资源分配
    - 提交和跟踪任务
    """
    
    def __init__(self, num_gpus_per_trial: float = 1.0, num_gpus_per_train: float = 1.0):
        self.num_gpus_per_trial = num_gpus_per_trial
        self.num_gpus_per_train = num_gpus_per_train
        self._ray = None
        self._remote_funcs = {}
    
    def init_ray(self, address: Optional[str] = None, **kwargs):
        """初始化 Ray"""
        self._ray = _check_ray()
        if not self._ray.is_initialized():
            self._ray.init(address=address, **kwargs)
        
        # 创建 remote functions
        self._create_remote_funcs()
    
    def _create_remote_funcs(self):
        """创建 Ray remote functions"""
        ray = self._ray
        
        # CPU tasks
        self._remote_funcs["run_split"] = ray.remote(num_gpus=0)(_run_split_impl)
        self._remote_funcs["update_state"] = ray.remote(num_gpus=0)(_update_state_impl)
        self._remote_funcs["aggregate_importance"] = ray.remote(num_gpus=0)(_aggregate_importance_impl)
        self._remote_funcs["update_state_from_bucket"] = ray.remote(num_gpus=0)(_update_state_from_bucket_impl)
        
        # GPU tasks
        self._remote_funcs["run_trial"] = ray.remote(num_gpus=self.num_gpus_per_trial)(_run_trial_impl)
        self._remote_funcs["run_final_train"] = ray.remote(num_gpus=self.num_gpus_per_train)(_run_final_train_impl)
    
    def run_split(
        self,
        task: SplitTask,
        global_ref: Any,
        state_ref: Any,
        config: ExperimentConfig,
        train_config: TrainConfig,
        method_config: Optional[Dict[str, Any]] = None,
        data_processor_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """提交 run_split 任务"""
        return self._remote_funcs["run_split"].remote(
            task, global_ref, state_ref, config, train_config, method_config, data_processor_config
        )
    
    def run_trial(
        self,
        trial_id: int,
        params: Dict[str, Any],
        views_ref: Any,
        train_config: TrainConfig,
        seed: int,
    ) -> Any:
        """提交 run_trial 任务"""
        return self._remote_funcs["run_trial"].remote(
            trial_id, params, views_ref, train_config, seed
        )
    
    def run_final_train(
        self,
        views_ref: Any,
        train_config: TrainConfig,
        save_dir: Path,
    ) -> Any:
        """提交 run_final_train 任务"""
        return self._remote_funcs["run_final_train"].remote(
            views_ref, train_config, save_dir
        )
    
    def update_state(
        self,
        state_ref: Any,
        result_ref: Any,
        config: StatePolicyConfig,
    ) -> Any:
        """提交 update_state 任务"""
        return self._remote_funcs["update_state"].remote(
            state_ref, result_ref, config
        )
    
    def aggregate_importance(
        self,
        result_refs: List[Any],
        config: StatePolicyConfig,
    ) -> Any:
        """提交 aggregate_importance 任务"""
        ray = self._ray
        # 先获取所有结果
        results = ray.get(result_refs)
        return self._remote_funcs["aggregate_importance"].remote(results, config)
    
    def update_state_from_bucket(
        self,
        state_ref: Any,
        result_refs: List[Any],
        config: StatePolicyConfig,
    ) -> Any:
        """提交 update_state_from_bucket 任务"""
        ray = self._ray
        results = ray.get(result_refs)
        return self._remote_funcs["update_state_from_bucket"].remote(
            state_ref, results, config
        )
    
    def put(self, obj: Any) -> Any:
        """将对象放入 Ray object store"""
        return self._ray.put(obj)
    
    def get(self, refs: Any) -> Any:
        """从 Ray object store 获取对象"""
        return self._ray.get(refs)
    
    def shutdown(self):
        """关闭 Ray"""
        if self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()


# =============================================================================
# 4. Local Execution (No Ray)
# =============================================================================

class LocalTaskManager:
    """
    本地任务管理器 (不使用 Ray)
    
    用于调试和小规模实验
    """
    
    def run_split(
        self,
        task: SplitTask,
        global_store: GlobalStore,
        rolling_state: Optional[RollingState],
        config: ExperimentConfig,
        train_config: TrainConfig,
        method_config: Optional[Dict[str, Any]] = None,
        data_processor_config: Optional[Dict[str, Any]] = None,
    ) -> SplitResult:
        """本地执行 run_split"""
        return _run_split_impl(task, global_store, rolling_state, config, train_config, method_config, data_processor_config)
    
    def update_state(
        self,
        prev_state: Optional[RollingState],
        split_result: SplitResult,
        config: StatePolicyConfig,
    ) -> RollingState:
        """本地执行 update_state"""
        return _update_state_impl(prev_state, split_result, config)
    
    def aggregate_importance(
        self,
        results: List[SplitResult],
        config: StatePolicyConfig,
    ) -> np.ndarray:
        """本地执行 aggregate_importance"""
        return _aggregate_importance_impl(results, config)
    
    def update_state_from_bucket(
        self,
        prev_state: Optional[RollingState],
        results: List[SplitResult],
        config: StatePolicyConfig,
    ) -> RollingState:
        """本地执行 update_state_from_bucket"""
        return _update_state_from_bucket_impl(prev_state, results, config)


# =============================================================================
# 5. Task Factory
# =============================================================================

def create_task_manager(
    use_ray: bool = True,
    num_gpus_per_trial: float = 1.0,
    num_gpus_per_train: float = 1.0,
    ray_address: Optional[str] = None,
    **ray_kwargs,
):
    """
    创建任务管理器
    
    Args:
        use_ray: 是否使用 Ray
        num_gpus_per_trial: 每个 trial 的 GPU 数
        num_gpus_per_train: 最终训练的 GPU 数
        ray_address: Ray 地址
        **ray_kwargs: 其他 Ray 初始化参数
        
    Returns:
        任务管理器实�?
    """
    if use_ray:
        manager = RayTaskManager(
            num_gpus_per_trial=num_gpus_per_trial,
            num_gpus_per_train=num_gpus_per_train,
        )
        manager.init_ray(address=ray_address, **ray_kwargs)
        return manager
    else:
        return LocalTaskManager()
