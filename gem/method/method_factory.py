"""
Method factory for assembling runtime components from Hydra configs.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from .base import BaseMethod
from .method_dataclasses import TrainConfig


class MethodFactory:
    @staticmethod
    def build(
        *,
        method_config: Optional[Mapping[str, Any]],
        train_config: TrainConfig,
        n_trials: int,
        parallel_trials: int,
        use_ray_tune: bool,
        base_seed: int,
        split_id: int,
    ) -> tuple[BaseMethod, TrainConfig]:
        config = MethodFactory._to_plain_mapping(method_config)
        model_config = config.get("model")
        resolved_train_config = MethodFactory._resolve_train_config(
            train_config,
            model_config,
        )

        adapter = None
        if config.get("adapter") is not None:
            adapter = instantiate(config["adapter"])

        trainer_overrides = {}
        if adapter is not None:
            trainer_overrides["adapter"] = adapter
        if model_config is not None:
            trainer_overrides["model_config"] = model_config
        trainer = instantiate(
            config["trainer"],
            **MethodFactory._filter_supported_overrides(config["trainer"], trainer_overrides),
        )
        evaluator = instantiate(config["evaluator"])
        importance_extractor = instantiate(config["importance_extractor"])

        tuner = None
        if n_trials > 0 and config.get("tuner") is not None:
            tuner_cfg = config["tuner"]
            tune_overrides = {
                "base_params": resolved_train_config.params,
                "n_trials": n_trials,
                "parallel_trials": parallel_trials,
                "use_ray_tune": use_ray_tune,
                "seed": base_seed + split_id,
            }
            tuner = instantiate(
                tuner_cfg,
                **MethodFactory._filter_supported_overrides(tuner_cfg, tune_overrides),
            )

        transform_pipeline = None
        if config.get("transform_pipeline") is not None:
            transform_pipeline = instantiate(config["transform_pipeline"])

        method = BaseMethod(
            transform_pipeline=transform_pipeline,
            adapter=adapter,
            trainer=trainer,
            evaluator=evaluator,
            importance_extractor=importance_extractor,
            tuner=tuner,
        )
        return method, resolved_train_config

    @staticmethod
    def _filter_supported_overrides(
        component_config: Any,
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        target = None
        if isinstance(component_config, Mapping):
            target = component_config.get("_target_")
        elif hasattr(component_config, "get"):
            target = component_config.get("_target_")

        if not target:
            return dict(overrides)

        try:
            cls = get_class(str(target))
            signature = inspect.signature(cls.__init__)
        except Exception:
            return dict(overrides)

        return {
            key: value
            for key, value in overrides.items()
            if key in signature.parameters
        }

    @staticmethod
    def _to_plain_mapping(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        if config is None:
            return {}
        if OmegaConf.is_config(config):
            resolved = OmegaConf.to_container(config, resolve=True)
            if isinstance(resolved, Mapping):
                return dict(resolved)
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        return {}

    @staticmethod
    def _extract_model_defaults(model_config: Optional[Any]) -> dict[str, Any]:
        if model_config is None:
            return {}
        plain_model = MethodFactory._to_plain_mapping(model_config)
        if not plain_model:
            return {}
        plain_model.pop("_target_", None)
        return plain_model

    @staticmethod
    def _resolve_train_config(
        train_config: TrainConfig,
        model_config: Optional[Any],
    ) -> TrainConfig:
        model_defaults = MethodFactory._extract_model_defaults(model_config)
        merged_params = dict(model_defaults)
        merged_params.update(dict(train_config.params or {}))
        return TrainConfig(
            params=merged_params,
            num_boost_round=train_config.num_boost_round,
            early_stopping_rounds=train_config.early_stopping_rounds,
            feval_names=list(train_config.feval_names),
            objective_name=train_config.objective_name,
            seed=train_config.seed,
            verbose_eval=train_config.verbose_eval,
            use_ray_trainer=train_config.use_ray_trainer,
        )
