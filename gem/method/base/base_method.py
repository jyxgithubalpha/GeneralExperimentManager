"""
Method runtime pipeline: tune -> train -> evaluate -> importance -> artifacts.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ...experiment.states import RollingState
from ...data.data_dataclasses import ProcessedViews, SplitViews
from ..method_dataclasses import MethodOutput, StateDelta, TrainConfig, TuneResult
from .base_adapter import BaseAdapter
from .base_evaluator import BaseEvaluator
from .base_importance_extractor import BaseImportanceExtractor
from .base_trainer import BaseTrainer
from .tuning import UnifiedTuner
from .transforms import BaseTransformPipeline


@dataclass(frozen=True)
class MethodComponents:
    trainer: BaseTrainer
    evaluator: BaseEvaluator
    importance_extractor: BaseImportanceExtractor
    adapter: Optional[BaseAdapter] = None
    tuner: Optional[UnifiedTuner] = None
    transform_pipeline: Optional[BaseTransformPipeline] = None


class BaseMethod:
    def __init__(
        self,
        framework: str = "lightgbm",
        transform_pipeline: Optional[BaseTransformPipeline] = None,
        adapter: Optional[BaseAdapter] = None,
        trainer: Optional[BaseTrainer] = None,
        evaluator: Optional[BaseEvaluator] = None,
        importance_extractor: Optional[BaseImportanceExtractor] = None,
        tuner: Optional[UnifiedTuner] = None,
        **_: Any,
    ):
        self.framework = framework
        self.components = MethodComponents(
            trainer=self._require_component("trainer", trainer),
            evaluator=self._require_component("evaluator", evaluator),
            importance_extractor=self._require_component(
                "importance_extractor",
                importance_extractor,
            ),
            adapter=adapter,
            tuner=tuner,
            transform_pipeline=transform_pipeline,
        )

        # Convenience attribute access.
        self.transform_pipeline = self.components.transform_pipeline
        self.adapter = self.components.adapter
        self.tuner = self.components.tuner
        self.trainer = self.components.trainer
        self.evaluator = self.components.evaluator
        self.importance_extractor = self.components.importance_extractor

    @staticmethod
    def _require_component(name: str, value: Any) -> Any:
        if value is None:
            raise ValueError(f"Method component '{name}' is required but not provided.")
        return value

    def run(
        self,
        views: Union[SplitViews, ProcessedViews],
        config: TrainConfig,
        do_tune: bool = True,
        save_dir: Optional[Path] = None,
        rolling_state: Optional["RollingState"] = None,
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> MethodOutput:
        processed_views = self._prepare_processed_views(views, rolling_state)

        best_params, tune_result = self._run_tuning(
            views=processed_views,
            config=config,
            do_tune=do_tune,
            rolling_state=rolling_state,
        )
        final_config = self._build_final_config(config, best_params)

        fit_result = self.trainer.fit(
            processed_views,
            final_config,
            mode="full",
            sample_weights=sample_weights,
        )
        metrics_eval = self.evaluator.evaluate(fit_result.model, processed_views)

        importance_vector, importance_df = self._extract_importance(
            model=fit_result.model,
            feature_names=processed_views.train.feature_names,
        )
        fit_result.feature_importance = importance_df

        model_artifacts = self._save_artifacts(
            model=fit_result.model,
            importance_df=importance_df,
            save_dir=save_dir,
        )

        feature_names_hash = processed_views.train.get_feature_names_hash()
        state_delta = self._build_state_delta(
            importance_vector=importance_vector,
            feature_names_hash=feature_names_hash,
            best_params=best_params,
            tune_result=tune_result,
        )

        return MethodOutput(
            best_params=best_params,
            metrics_eval=metrics_eval,
            importance_vector=importance_vector,
            feature_names_hash=feature_names_hash,
            tune_result=tune_result,
            fit_result=fit_result,
            state_delta=state_delta,
            model_artifacts=model_artifacts,
        )

    def _prepare_processed_views(
        self,
        views: Union[SplitViews, ProcessedViews],
        rolling_state: Optional["RollingState"],
    ) -> ProcessedViews:
        if isinstance(views, ProcessedViews):
            return views

        transform_pipeline = self.transform_pipeline
        if transform_pipeline is not None:
            if not hasattr(transform_pipeline, "fit_transform_views"):
                raise TypeError(
                    "transform_pipeline must implement fit_transform_views(views, rolling_state=...)."
                )
            processed_views, _ = transform_pipeline.fit_transform_views(
                views,
                rolling_state=rolling_state,
            )
            return processed_views

        return ProcessedViews(
            train=views.train,
            val=views.val,
            test=views.test,
            split_spec=views.split_spec,
        )

    def _run_tuning(
        self,
        views: ProcessedViews,
        config: TrainConfig,
        do_tune: bool,
        rolling_state: Optional["RollingState"],
    ) -> Tuple[Dict[str, Any], Optional[TuneResult]]:
        best_params = dict(config.params)
        if not do_tune or self.tuner is None:
            return best_params, None

        tune_result = self.tuner.tune(
            views=views,
            trainer=self.trainer,
            config=config,
            rolling_state=rolling_state,
        )
        if tune_result.best_params:
            best_params = dict(tune_result.best_params)
        return best_params, tune_result

    @staticmethod
    def _build_final_config(config: TrainConfig, best_params: Dict[str, Any]) -> TrainConfig:
        return TrainConfig(
            params=dict(best_params),
            num_boost_round=config.num_boost_round,
            early_stopping_rounds=config.early_stopping_rounds,
            feval_names=list(config.feval_names),
            objective_name=config.objective_name,
            seed=config.seed,
            verbose_eval=config.verbose_eval,
            use_ray_trainer=config.use_ray_trainer,
        )

    @staticmethod
    def _save_artifacts(model: Any, importance_df, save_dir: Optional[Path]) -> Dict[str, Path]:
        artifacts: Dict[str, Path] = {}
        if save_dir is None:
            return artifacts

        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / "model.txt"
        if hasattr(model, "save_model"):
            model.save_model(str(model_path))
            artifacts["model"] = model_path

        if importance_df is not None:
            importance_path = path / "feature_importance.csv"
            importance_df.write_csv(str(importance_path))
            artifacts["importance"] = importance_path

        return artifacts

    @staticmethod
    def _build_state_delta(
        importance_vector: np.ndarray,
        feature_names_hash: str,
        best_params: Dict[str, Any],
        tune_result: Optional[TuneResult],
    ) -> StateDelta:
        return StateDelta(
            importance_vector=importance_vector,
            feature_names_hash=feature_names_hash,
            best_params=best_params,
            best_objective=tune_result.best_value if tune_result else None,
        )

    def _extract_importance(self, model: Any, feature_names) -> Tuple[np.ndarray, Any]:
        importance_vector, importance_df = self.importance_extractor.extract(
            model,
            feature_names,
        )
        if importance_vector.shape[0] != len(feature_names):
            raise ValueError(
                "Importance vector size does not match feature names size: "
                f"{importance_vector.shape[0]} vs {len(feature_names)}"
            )
        return importance_vector, importance_df
