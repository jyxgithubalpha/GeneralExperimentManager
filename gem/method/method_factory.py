"""
Method factory for assembling runtime components from Hydra configs.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from hydra.utils import get_class, instantiate

from .base import BaseMethod
from .method_dataclasses import TrainConfig


class MethodFactory:
    @staticmethod
    def build(
        *,
        method_config: Optional[Mapping[str, Any]],
        train_config: TrainConfig,
        n_trials: int,
        trial_timeout: Optional[int],
        parallel_trials: int,
        use_ray_tune: bool,
        base_seed: int,
        split_id: int,
        adapter_config: Optional[Any] = None,
        transform_config: Optional[Any] = None,
    ) -> BaseMethod:
        config = dict(method_config or {})

        required_keys = ["trainer", "evaluator", "importance_extractor"]
        missing = [key for key in required_keys if config.get(key) is None]
        if missing:
            raise ValueError(
                f"Missing required method components in method_config: {missing}"
            )

        adapter = None
        if adapter_config is not None:
            adapter = instantiate(adapter_config)
        elif config.get("adapter") is not None:
            adapter = instantiate(config["adapter"])

        trainer_overrides = {}
        if adapter is not None:
            trainer_overrides["adapter"] = adapter
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
                "base_params": train_config.params,
                "n_trials": n_trials,
                "timeout": trial_timeout,
                "parallel_trials": parallel_trials,
                "use_ray_tune": use_ray_tune,
                "seed": base_seed + split_id,
            }
            tuner = instantiate(
                tuner_cfg,
                **MethodFactory._filter_supported_overrides(tuner_cfg, tune_overrides),
            )

        transform_pipeline = None
        if transform_config is not None:
            transform_pipeline = instantiate(transform_config)
        elif config.get("transform_pipeline") is not None:
            transform_pipeline = instantiate(config["transform_pipeline"])

        return BaseMethod(
            transform_pipeline=transform_pipeline,
            adapter=adapter,
            trainer=trainer,
            evaluator=evaluator,
            importance_extractor=importance_extractor,
            tuner=tuner,
        )

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
