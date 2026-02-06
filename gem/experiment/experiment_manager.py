"""
ExperimentManager - 实验管理器

负责:
- 构建执行计划 (并行/串行/bucket)
- 用 Ray 调度
- 聚合结果并输出报告

ExperimentManager.run() 工作步骤:
1. ray.init(...)
2. splitspec_list, dr_start, dr_end = split_generator.generate(...)
3. global_store = datamodule.prepare_global_store(dr_start, dr_end, base_spec)
4. global_ref = ray.put(global_store)
5. 构造 tasks = [SplitTask(...)] 并按 test_start 排序
6. 初始化 state_ref = ray.put(init_state)
7. 按 state_policy 构建 DAG 并执行
8. 聚合 summary、写全局报告
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

from ..data.data_module import DataModule
from ..data.split_generator import SplitGenerator
from ..data.data_dataclasses import GlobalStore, SplitSpec
from .state_dataclasses import (
    RollingState,
    StatePolicyMode,
    ResourceRequest,
    ExperimentConfig,
    SplitTask,
    SplitResult,
)
from .state_policy import BucketManager
from ..method.training_dataclasses import TrainConfig



class ExperimentManager:
    def __init__(
        self,
        split_generator: SplitGenerator,
        data_module: DataModule,
        train_config: TrainConfig,
        experiment_config: ExperimentConfig,
        method_config: Optional[Dict] = None,
        data_processor_config: Optional[Dict] = None,
    ):
        self.experiment_config = experiment_config
        self.split_generator = split_generator
        self.data_module = data_module
        self.train_config = train_config
        self.method_config = method_config
        self.data_processor_config = data_processor_config
        
        # Results
        self._results: Dict[int, SplitResult] = {}
        self._global_store: Optional[GlobalStore] = None
        self._splitspec_list: Optional[List[SplitSpec]] = None
        self._start_time: Optional[float] = None
        self._use_ray: bool = self.experiment_config.use_ray
    
    @property
    def use_ray(self) -> bool:
        return self._use_ray
    
    def run(self) -> Dict[int, SplitResult]:
        self._start_time = time.time()
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {self.experiment_config.name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        # 1. Generate splits
        print("[1/6] Generating splits...")
        gen_output = self.split_generator.generate()
        splitspec_list = gen_output.splitspec_list
        self._splitspec_list = splitspec_list
        dr_start = gen_output.date_start
        dr_end = gen_output.date_end
        print(f"  - Generated {len(splitspec_list)} splits")
        
        # 2. Prepare global store
        print("\n[2/6] Preparing global store...")
        self._global_store = self.data_module.prepare_global_store(dr_start, dr_end)
        print(f"  - Samples: {self._global_store.n_samples}")
        print(f"  - Features: {self._global_store.n_features}")
        
        # 3. Build tasks
        print("\n[3/6] Building tasks...")
        tasks = self._build_tasks(splitspec_list)
        tasks = sorted(tasks, key=lambda t: t.splitspec.test_date_list[0] if t.splitspec.test_date_list else 0)
        print(f"  - Built {len(tasks)} tasks")
        
        # 4. Initialize state
        print("\n[4/6] Initializing state...")
        init_state = RollingState()
        
        # 5. Execute based on policy
        print(f"\n[5/6] Executing with policy: {self.experiment_config.state_policy.mode}")
        
        if self.use_ray:
            results = self._run_with_ray(tasks, init_state)
        else:
            results = self._run_local(tasks, init_state)
        
        self._results = {r.split_id: r for r in results}
        
        # 6. Generate report
        print("\n[6/6] Generating report...")
        self._generate_report(output_dir)
        
        elapsed = time.time() - self._start_time
        print(f"\n{'='*60}")
        print(f"Experiment completed in {elapsed:.1f}s")
        print(f"{'='*60}\n")
        
        return self._results
    
    def _build_tasks(self, splitspec_list: List[SplitSpec]) -> List[SplitTask]:
        """构建任务列表"""
        tasks = []
        for i, spec in enumerate(splitspec_list):
            task = SplitTask(
                split_id=spec.split_id,
                splitspec=spec,
                seed=self.experiment_config.seed + i,
                resource_request=self.experiment_config.resource_request,
                train_config=self.train_config,
            )
            tasks.append(task)
        return tasks
    
    def _run_local(
        self,
        tasks: List[SplitTask],
        init_state: RollingState,
    ) -> List[SplitResult]:
        """本地执行 (无 Ray)"""
        from .ray_tasks import LocalTaskManager
        
        manager = LocalTaskManager()
        policy_config = self.experiment_config.state_policy
        mode = policy_config.mode
        
        # 创建执行计划
        bucket_manager = BucketManager(bucket_fn=policy_config.bucket_fn)
        execution_plan = bucket_manager.create_execution_plan(
            [t.splitspec for t in tasks], mode
        )
        
        # 创建 task 索引
        task_map = {t.splitspec.split_id: t for t in tasks}
        
        results = []
        current_state = init_state
        
        if mode == StatePolicyMode.NONE.value:
            # 全并行 (本地串行执行)
            print(f"  - Mode: NONE (parallel conceptually, serial locally)")
            for i, task in enumerate(tasks):
                print(f"    Running split {task.split_id} ({i+1}/{len(tasks)})")
                result = manager.run_split(
                    task, self._global_store, None, self.experiment_config, self.train_config, self.method_config, self.data_processor_config
                )
                results.append(result)
                self._print_split_result(result)
        
        elif mode == StatePolicyMode.PER_SPLIT.value:
            # 严格串行
            print(f"  - Mode: PER_SPLIT (strict serial)")
            for i, batch in enumerate(execution_plan):
                for spec in batch:
                    task = task_map[spec.split_id]
                    print(f"    Running split {task.split_id}")
                    result = manager.run_split(
                        task, self._global_store, current_state, self.experiment_config, self.train_config, self.method_config, self.data_processor_config
                    )
                    results.append(result)
                    self._print_split_result(result)
                    
                    # 更新状态
                    current_state = manager.update_state(
                        current_state, result, policy_config
                    )
        
        elif mode == StatePolicyMode.BUCKET.value:
            # Bucket 内并行 (本地串行)，Bucket 间串行
            print(f"  - Mode: BUCKET")
            for bucket_idx, batch in enumerate(execution_plan):
                print(f"    Bucket {bucket_idx + 1}/{len(execution_plan)} ({len(batch)} splits)")
                
                bucket_results = []
                for spec in batch:
                    task = task_map[spec.split_id]
                    print(f"      Running split {task.split_id}")
                    result = manager.run_split(
                        task, self._global_store, current_state, self.experiment_config, self.train_config, self.method_config, self.data_processor_config
                    )
                    bucket_results.append(result)
                    results.append(result)
                    self._print_split_result(result)
                
                # Bucket 结束后更新状态
                current_state = manager.update_state_from_bucket(
                    current_state, bucket_results, policy_config
                )
        
        return results
    
    def _run_with_ray(
        self,
        tasks: List[SplitTask],
        init_state: RollingState,
    ) -> List[SplitResult]:
        """使用 Ray 执行"""
        from .ray_tasks import RayTaskManager
        
        manager = RayTaskManager(
            num_gpus_per_trial=0,
            num_gpus_per_train=0,
        )
        manager.init_ray(
            address=self.experiment_config.ray_address,
            num_cpus=self.experiment_config.num_cpus,
            num_gpus=self.experiment_config.num_gpus,
        )
        
        try:
            policy_config = self.experiment_config.state_policy
            mode = policy_config.mode
            
            # Put global store in object store
            global_ref = manager.put(self._global_store)
            
            # 创建执行计划
            bucket_manager = BucketManager(bucket_fn=policy_config.bucket_fn)
            execution_plan = bucket_manager.create_execution_plan(
                [t.splitspec for t in tasks], mode
            )
            
            task_map = {t.splitspec.split_id: t for t in tasks}
            result_refs = []
            
            if mode == StatePolicyMode.NONE.value:
                # 全并行
                print(f"  - Mode: NONE (fully parallel)")
                state_ref = manager.put(None)
                
                refs = []
                for task in tasks:
                    ref = manager.run_split(
                        task, global_ref, state_ref, self.experiment_config, self.train_config, self.method_config, self.data_processor_config
                    )
                    refs.append(ref)
                
                results = manager.get(refs)
            
            elif mode == StatePolicyMode.PER_SPLIT.value:
                # 严格串行
                print(f"  - Mode: PER_SPLIT (strict serial)")
                state_ref = manager.put(init_state)
                
                results = []
                for batch in execution_plan:
                    for spec in batch:
                        task = task_map[spec.split_id]
                        print(f"    Running split {task.split_id}")
                        
                        result_ref = manager.run_split(
                            task, global_ref, state_ref, self.experiment_config, self.train_config, self.method_config, self.data_processor_config
                        )
                        result = manager.get(result_ref)
                        results.append(result)
                        self._print_split_result(result)
                        
                        # 更新状态
                        state_ref = manager.update_state(
                            state_ref, result_ref, policy_config
                        )
            
            elif mode == StatePolicyMode.BUCKET.value:
                # Bucket 模式
                print(f"  - Mode: BUCKET")
                state_ref = manager.put(init_state)
                
                results = []
                for bucket_idx, batch in enumerate(execution_plan):
                    print(f"    Bucket {bucket_idx + 1}/{len(execution_plan)}")
                    
                    # Bucket 内并行
                    refs = []
                    for spec in batch:
                        task = task_map[spec.split_id]
                        ref = manager.run_split(
                            task, global_ref, state_ref, self.experiment_config, self.train_config, self.method_config, self.data_processor_config
                        )
                        refs.append(ref)
                    
                    bucket_results = manager.get(refs)
                    results.extend(bucket_results)
                    
                    for r in bucket_results:
                        self._print_split_result(r)
                    
                    # Bucket 间更新状态
                    state_ref = manager.update_state_from_bucket(
                        state_ref, refs, policy_config
                    )
            
            else:
                results = []
            
            return results
            
        finally:
            manager.shutdown()
    
    def _print_split_result(self, result: SplitResult):
        """打印 split 结果"""
        if result.skipped:
            print(f"      [SKIPPED] {result.skip_reason}")
        elif result.metrics:
            if "error" in result.metrics:
                error_msg = str(result.metrics["error"])[:100]
                print(f"      [ERROR] {error_msg}")
            else:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in list(result.metrics.items())[:3]
                )
                print(f"      [OK] {metrics_str}")
        else:
            print(f"      [OK] No metrics")
    
    def _generate_report(self, output_dir: Path):
        """生成实验报告"""
        # Summary DataFrame
        rows = []
        for split_id, result in self._results.items():
            row = {"split_id": split_id, "skipped": result.skipped, "skip_reason": result.skip_reason}
            if result.metrics:
                row.update(result.metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows).sort_values("split_id")
        
        # Save CSV
        csv_path = output_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"  - Saved: {csv_path}")
        
        # Save JSON config
        config_path = output_dir / "config.json"
        config_dict = {
            "name": self.experiment_config.name,
            "seed": self.experiment_config.seed,
            "n_trials": self.experiment_config.n_trials,
            "state_policy_mode": self.experiment_config.state_policy.mode,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self._start_time,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"  - Saved: {config_path}")
        
        # Print summary
        n_skipped = sum(1 for r in self._results.values() if r.skipped)
        n_success = len(self._results) - n_skipped
        print(f"\n  Splits: {n_success} success, {n_skipped} skipped")
        
        metric_cols = [c for c in df.columns if c not in ("split_id", "skipped", "skip_reason")]
        if metric_cols:
            print("\n  Metrics Summary:")
            for col in metric_cols[:5]:
                if col in df.columns:
                    valid_vals = df[col].dropna()
                    if len(valid_vals) > 0:
                        mean_val = valid_vals.mean()
                        std_val = valid_vals.std()
                        print(f"    {col}: {mean_val:.4f} ± {std_val:.4f}")
    
    @property
    def results(self) -> Dict[int, SplitResult]:
        return self._results
    
    @property
    def global_store(self) -> Optional[GlobalStore]:
        return self._global_store
    
    @property
    def splitspec_list(self) -> Optional[List[SplitSpec]]:
        return self._splitspec_list
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        if self._global_store:
            return self._global_store.feature_name_list
        return None