"""
Transform Pipeline - Pipeline for chaining transforms.
"""


from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ....data.data_dataclasses import ProcessedViews, SplitView, SplitViews
from ....experiment.states import RollingState
from ...method_dataclasses import RayDataBundle, RayDataViews, TransformState, TransformStats
from .base import BaseTransform, TransformContext, extract_date_keys
from .feature_weight import FeatureWeightTransform
from .stats import StatsCalculator


class FittedTransformPipeline:
    """
    Fitted Transform Pipeline.
    
    Holds fitted transforms, can only transform but not fit.
    """
    
    def __init__(
        self,
        transforms: List[BaseTransform],
        context: TransformContext,
        feature_names: Optional[List[str]] = None,
    ):
        self.transforms = transforms
        self._context = context
        self._feature_names = feature_names
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transforms."""
        X_curr, y_curr = X.copy(), y.copy()
        for transform in self.transforms:
            X_curr, y_curr = transform.transform(X_curr, y_curr, keys)
        return X_curr, y_curr
    
    def transform_views(self, views: "SplitViews") -> "SplitViews":
        """
        Apply transforms to all views of SplitViews.
        
        Args:
            views: Input views
            
        Returns:
            Transformed SplitViews
        """
        from ....data.data_dataclasses import SplitView, SplitViews
        
        train_keys = extract_date_keys(views.train.keys)
        val_keys = extract_date_keys(views.val.keys)
        test_keys = extract_date_keys(views.test.keys)
        
        train_X, train_y = self.transform(views.train.X, views.train.y, train_keys)
        val_X, val_y = self.transform(views.val.X, views.val.y, val_keys)
        test_X, test_y = self.transform(views.test.X, views.test.y, test_keys)
        
        feature_names = self._feature_names
        for transform in self.transforms:
            if isinstance(transform, FeatureWeightTransform) and transform.selected_feature_indices is not None:
                if feature_names is not None:
                    feature_names = [feature_names[i] for i in transform.selected_feature_indices]
        
        return SplitViews(
            train=SplitView(
                indices=views.train.indices,
                X=train_X,
                y=train_y,
                keys=views.train.keys,
                feature_names=feature_names,
                label_names=views.train.label_names,
                extra=views.train.extra,
            ),
            val=SplitView(
                indices=views.val.indices,
                X=val_X,
                y=val_y,
                keys=views.val.keys,
                feature_names=feature_names,
                label_names=views.val.label_names,
                extra=views.val.extra,
            ),
            test=SplitView(
                indices=views.test.indices,
                X=test_X,
                y=test_y,
                keys=views.test.keys,
                feature_names=feature_names,
                label_names=views.test.label_names,
                extra=views.test.extra,
            ),
            split_spec=views.split_spec,
        )
    
    @property
    def feature_names(self) -> Optional[List[str]]:
        return self._feature_names


class BaseTransformPipeline:
    """
    Data processor.
    
    Manages pipeline of multiple transforms, supports passing external parameters through context.
    """
    
    def __init__(self, transforms: Optional[List[BaseTransform]] = None):
        self.transforms = transforms or []
        self._context: TransformContext = {}
    
    def set_context(self, context: TransformContext) -> "BaseTransformPipeline":
        """Set context and pass to all transforms."""
        self._context = context
        for transform in self.transforms:
            transform.set_context(context)
        return self
    
    def add_transform(self, transform: BaseTransform) -> "BaseTransformPipeline":
        """Add transform."""
        transform.set_context(self._context)
        self.transforms.append(transform)
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> "BaseTransformPipeline":
        """Fit all transforms."""
        X_curr, y_curr = X.copy(), y.copy()
        for transform in self.transforms:
            transform.fit(X_curr, y_curr, keys)
            X_curr, y_curr = transform.transform(X_curr, y_curr, keys)
        return self
    
    def transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transforms."""
        X_curr, y_curr = X.copy(), y.copy()
        for transform in self.transforms:
            X_curr, y_curr = transform.transform(X_curr, y_curr, keys)
        return X_curr, y_curr
    
    def fit_transform_arrays(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform."""
        self.fit(X, y, keys)
        return self.transform(X, y, keys)
    
    def inverse_transform(self, X: np.ndarray, y: np.ndarray, keys: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse transform (reverse order)."""
        X_curr, y_curr = X.copy(), y.copy()
        for transform in reversed(self.transforms):
            X_curr, y_curr = transform.inverse_transform(X_curr, y_curr, keys)
        return X_curr, y_curr
    
    def fit_on_train(
        self,
        train_view: "SplitView",
        rolling_state: Optional["RollingState"] = None,
    ) -> FittedTransformPipeline:
        """
        Fit on train view, return FittedTransformPipeline.
        
        Args:
            train_view: Training view
            rolling_state: Rolling state (for getting context)
            
        Returns:
            FittedTransformPipeline
        """
        context = {}
        if rolling_state is not None:
            context = rolling_state.to_transform_context()
        self.set_context(context)
        
        train_keys = extract_date_keys(train_view.keys)
        self.fit(train_view.X, train_view.y, train_keys)
        
        return FittedTransformPipeline(
            transforms=self.transforms,
            context=self._context,
            feature_names=train_view.feature_names,
        )
    
    def process_views(
        self,
        views: "SplitViews",
        context: Optional[TransformContext] = None,
    ) -> "SplitViews":
        """
        Process SplitViews.
        
        Args:
            views: Input views
            context: Optional context parameters, from RollingState.to_transform_context()
        
        Fit on train, then transform train/val/test.
        """
        from ....data.data_dataclasses import SplitView, SplitViews
        
        if context is not None:
            self.set_context(context)
        
        train_keys = extract_date_keys(views.train.keys)
        val_keys = extract_date_keys(views.val.keys)
        test_keys = extract_date_keys(views.test.keys)
        
        self.fit(views.train.X, views.train.y, train_keys)
        
        train_X, train_y = self.transform(views.train.X, views.train.y, train_keys)
        val_X, val_y = self.transform(views.val.X, views.val.y, val_keys)
        test_X, test_y = self.transform(views.test.X, views.test.y, test_keys)
        
        feature_names = views.train.feature_names
        for transform in self.transforms:
            if isinstance(transform, FeatureWeightTransform) and transform.selected_feature_indices is not None:
                if feature_names is not None:
                    feature_names = [feature_names[i] for i in transform.selected_feature_indices]
        
        return SplitViews(
            train=SplitView(
                indices=views.train.indices,
                X=train_X,
                y=train_y,
                keys=views.train.keys,
                feature_names=feature_names,
                label_names=views.train.label_names,
                extra=views.train.extra,
            ),
            val=SplitView(
                indices=views.val.indices,
                X=val_X,
                y=val_y,
                keys=views.val.keys,
                feature_names=feature_names,
                label_names=views.val.label_names,
                extra=views.val.extra,
            ),
            test=SplitView(
                indices=views.test.indices,
                X=test_X,
                y=test_y,
                keys=views.test.keys,
                feature_names=feature_names,
                label_names=views.test.label_names,
                extra=views.test.extra,
            ),
            split_spec=views.split_spec,
        )
    
    def fit_transform_views(
        self,
        views: "SplitViews",
        rolling_state: Optional["RollingState"] = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> Tuple["ProcessedViews", TransformStats]:
        """
        Complete fit + transform workflow, returns ProcessedViews and TransformStats.
        
        Args:
            views: Input views
            rolling_state: Rolling state (for getting context)
            lower_quantile: Lower quantile threshold
            upper_quantile: Upper quantile threshold
            
        Returns:
            (ProcessedViews, TransformStats)
        """
        from ....data.data_dataclasses import ProcessedViews
        
        context = {}
        if rolling_state is not None:
            context = rolling_state.to_transform_context()
        
        stats = StatsCalculator.compute(
            X_train=views.train.X,
            y_train=views.train.y,
            X_val=views.val.X,
            y_val=views.val.y,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        
        transformed_views = self.process_views(views, context)
        
        processed = ProcessedViews(
            train=transformed_views.train,
            val=transformed_views.val,
            test=transformed_views.test,
            split_spec=views.split_spec,
            transform_state=TransformState(
                stats=self._collect_all_stats(),
                transform_stats=stats,
            ),
        )
        
        return processed, stats

    def _collect_all_stats(self) -> Dict[str, Any]:
        """Collect statistics from all transforms."""
        all_stats = {}
        for i, transform in enumerate(self.transforms):
            if transform.state is not None:
                all_stats[f"transform_{i}_{type(transform).__name__}"] = transform.state.stats
        return all_stats
    
    def to_ray_data_views(
        self,
        views: "SplitViews",
        rolling_state: Optional["RollingState"] = None,
        include_sample_weight: bool = True,
    ) -> Tuple[RayDataViews, TransformStats]:
        """
        Complete workflow: transform -> convert to RayDataViews.
        
        Args:
            views: Input views
            rolling_state: Rolling state
            include_sample_weight: Whether to include sample weights
            
        Returns:
            (RayDataViews, TransformStats)
        """
        from ....data.data_dataclasses import SplitView
        
        processed, stats = self.fit_transform_views(views, rolling_state)
        
        sample_weight_train = None
        sample_weight_val = None
        sample_weight_test = None
        
        if include_sample_weight and rolling_state is not None:
            from ....experiment.states import SampleWeightState
            weight_state = rolling_state.get_state(SampleWeightState)
            if weight_state is not None:
                sample_weight_train = weight_state.get_weights_for_view(processed.train)
                sample_weight_val = weight_state.get_weights_for_view(processed.val)
                sample_weight_test = weight_state.get_weights_for_view(processed.test)
        
        def _to_bundle(view: "SplitView", weight: Optional[np.ndarray]) -> RayDataBundle:
            keys_arr = None
            if view.keys is not None:
                keys_arr = view.keys.to_numpy()
            return RayDataBundle(
                X=view.X,
                y=view.y,
                keys=keys_arr,
                sample_weight=weight,
                feature_names=view.feature_names,
                label_names=view.label_names,
            )
        
        ray_views = RayDataViews(
            train=_to_bundle(processed.train, sample_weight_train),
            val=_to_bundle(processed.val, sample_weight_val),
            test=_to_bundle(processed.test, sample_weight_test),
            transform_state=processed.transform_state,
            transform_stats=stats,
        )
        
        return ray_views, stats
