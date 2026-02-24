# Method Config Guide

## Layer Responsibilities

- `model/*`: model defaults and model hyperparameters (single source of truth).
- `train_config/*`: training process controls (`num_boost_round`, `early_stopping_rounds`, `feval_names`, etc.).
- `trainer/*`: trainer runtime behavior (device, epochs, batch size, optimizer-related runtime flags).
- `method/*.yaml`: method assembly entrypoint, chooses config groups and defines method-specific overrides.
- `experiment/*`: experiment runtime policy (`n_trials`, `use_ray`, `state_policy`, `visualization`, resources).

## Parameter Priority

`model defaults < train_config.params < tuner sampled params`

- Model defaults come from `cfg.method.model`.
- `cfg.method.train_config.params` overrides model defaults.
- Tuner sampled params override both during tuning/final fit.

## Runtime Wiring

- `MethodFactory` merges `model` defaults into `TrainConfig.params`.
- If trainer supports `model_config`, `MethodFactory` injects `cfg.method.model` directly.
- `SplitRunner` uses resolved merged `TrainConfig` for both tuning and final training.

## Visualization Metric Names

- `experiment.visualization.metric_names` accepts daily metric ids only.
- Supported examples:
- `daily_ic`
- `daily_icir_expanding`
- `daily_top_ret`
- `daily_top_ret_std`
- `daily_top_ret_relative_improve_pct`
- `daily_model_benchmark_corr`
- Old metric names (for example `pearsonr_ic`, `top_ret`) are no longer mapped and now fail fast.

## Minimal Template for a New Method

```yaml
# gem/configs/method/new_method.yaml
defaults:
  - model: new_method
  - train_config: sklearn
  - trainer: sklearn
  - evaluator: regression
  - importance_extractor: sklearn
  - transform_pipeline: defaults
  - _self_

name: new_method
```

```yaml
# gem/configs/method/model/new_method.yaml
_target_: sklearn.linear_model.Ridge
alpha: 1.0
```

## Recommended Runtime Profiles

- `method=lgbm experiment=gbdt_cpu`
- `method=xgb experiment=gbdt_gpu`
- `method=cat experiment=gbdt_gpu`
- `method=ridge experiment=sklearn`
- `method=mlp experiment=torch`

## Startup Validation Rules

- `method.name` must be present and non-empty.
- `method.trainer`, `method.evaluator`, and `method.importance_extractor` must be configured.
- `experiment.n_trials > 0` requires `method.tuner`.
- `experiment.output_dir` is normalized to `Path` in `ExperimentConfig.__post_init__`.

## Method/Experiment Constraints

- Use `experiment=gbdt_cpu` or `experiment=gbdt_gpu` with methods that define a tuner (`lgbm`, `xgb`, `cat`).
- `experiment=sklearn` and `experiment=torch` default to `n_trials=0` and run without tuner.
