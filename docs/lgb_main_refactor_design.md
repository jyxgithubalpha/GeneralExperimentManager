# GEM Main Path Refactor (LGBM Default, No Compatibility Layer)

## 1. Architecture

### 1.1 Runtime Main Chain

```text
gem.run.main
  -> build_runtime(cfg) instantiate split_generator / data_module / experiment / train_config
  -> ExperimentManager.run
      -> split_generator.generate
      -> data_module.prepare_global_store
      -> build SplitTask list
      -> execute DAG (none / per_split / bucket) via LocalExecutor or RayExecutor
      -> aggregate SplitResult -> results_summary.csv + config.json
  -> optional post-run visualization plugin (disabled by default)
```

### 1.2 Experiment Config Shape (Base + Overlay)

```text
gem/configs/experiment/base.yaml
  -> shared defaults (_target_, output_dir, state_policy, visualization, resources)
gem/configs/experiment/{gbdt_cpu,gbdt_gpu,sklearn,torch}.yaml
  -> profile-specific overrides only
```

### 1.3 Data Flow

```text
SourceSpec (layout/date_col/code_col/pivot/index_col/value_col/rename_map)
  -> FeatherReader.read
      -> _process_dataframe(layout-rule driven)
      -> _standardize_columns(date/code normalization)
      -> date filter + concat batches
  -> single-source preprocess pipelines
  -> multi-source preprocess pipeline
  -> GlobalDataAssembler -> GlobalStore
```

### 1.4 State Flow

```text
RollingState
  -> to_transform_context()
  -> BaseTransformPipeline.fit_transform_views(...)
  -> Method.run(tune/train/evaluate/extract importance)
  -> SplitResult.state_delta
  -> DynamicTaskDAG applies state policy (none/per_split/bucket) across splits
```

## 2. File-Level Change List

### 2.1 Updated

- `gem/run.py`
- `gem/experiment/experiment_dataclasses.py`
- `gem/experiment/experiment_manager.py`
- `gem/experiment/split_runner.py`
- `gem/method/__init__.py`
- `gem/method/base/base_method.py`
- `gem/method/base/base_transform.py`
- `gem/method/lgb/lgb_evaluator.py`
- `gem/data/data_dataclasses.py`
- `gem/data/data_reader.py`
- `gem/configs/config.yaml`
- `gem/configs/method/lgbm.yaml`
- `gem/configs/method/evaluator/regression.yaml`
- `gem/configs/method/README.md`
- `gem/configs/experiment/gbdt_cpu.yaml`
- `gem/configs/experiment/gbdt_gpu.yaml`
- `gem/configs/experiment/sklearn.yaml`
- `gem/configs/experiment/torch.yaml`
- `gem/configs/experiment/base.yaml`
- `gem/configs/datamodule/sourcespec_dict/defaults.yaml`
- `gem/configs/datamodule/sourcespec_dict/SourceSpec/*.yaml`

### 2.2 Added

- `tests/test_method_config_refactor.py`
- `docs/lgb_main_refactor_design.md`
- `README.md`
- `requirements.txt`

### 2.3 Legacy / Not Main Path

- Non-LGBM frameworks (`xgb/cat/sklearn/torch`) are retained in source/config and can be selected by Hydra `method=...`.
- Legacy extensibility subsystem has been removed from the active codebase.

## 3. Hydra Config Migration Table

| Old path / behavior | New path / behavior |
|---|---|
| `defaults: method: lgb_main` in `gem/configs/config.yaml` | `defaults: method: lgbm` |
| `method/evaluator: lgbm -> gem.method.lgb.LightGBMEvaluator` | `method/evaluator: regression -> gem.method.common.RegressionEvaluator` |
| visualization config under method | top-level `experiment.visualization.*` |
| source-name hardcoded reader branches (`fac/ret/score/...`) | SourceSpec rule-driven reader via `layout/pivot/...` |
| sample weight via mixed method names | unified `SampleWeightState.get_weights_for_view(view)` |

## 4. Test Matrix and Acceptance

### 4.1 Unit

Command:

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected:

- `base_transform` key order invariance test passes.
- `SampleWeightState.get_weights_for_view` shape/value test passes.
- `FeatherReader` layout conversion and dtype checks pass.
- `MethodFactory` builds `lgbm` without registry import.

### 4.2 Integration / Smoke (Local)

Command:

```bash
.venv\Scripts\python.exe -m gem.run experiment.n_trials=0 experiment.use_ray=false experiment.visualization.enabled=false
```

Expected:

- Full split run completes.
- Outputs include `results_summary.csv`, `config.json`, per-split `model.txt`, `feature_importance.csv`.
- Metrics columns include prefixed keys (`train_*`, `val_*`, `test_*`).

### 4.3 Integration / Smoke (Ray)

Command:

```bash
.venv\Scripts\python.exe -m gem.run experiment.n_trials=0 experiment.use_ray=true experiment.visualization.enabled=false splitgenerator.test_date_start=20241226 splitgenerator.test_date_end=20241231 splitgenerator.train_len=5 splitgenerator.val_len=2 splitgenerator.test_len=2 splitgenerator.gap=1 splitgenerator.step=2
```

Expected:

- DAG execution succeeds for runnable splits.
- State passing across DAG tasks works under Ray executor.
- If Ray dashboard permission warning appears on Windows, core task execution can still pass.
