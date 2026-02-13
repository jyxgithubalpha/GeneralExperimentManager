"""
Legacy component registry.

Main runtime path uses Hydra-driven ``MethodFactory`` directly.
This module is retained for legacy/manual usage only.
"""



from typing import Any, Callable, Dict, List, Optional, Type

from .base import (
    BaseTransformPipeline,
    BaseEvaluator,
    BaseImportanceExtractor,
    BaseParamSpace,
    BaseTrainer,
    BaseTuner,
    BaseAdapter,
)


class ComponentRegistry:
    """
    Component registry
    
    Uses decorators or explicit registration methods to register components
    
    Example:
        # Use decorator registration
        @ComponentRegistry.register_trainer("lightgbm")
        class LightGBMTrainer(BaseTrainer):
            ...
        
        # Explicit registration
        ComponentRegistry.register("trainer", "xgboost", XGBoostTrainer)
        
        # Create component
        trainer = ComponentRegistry.create("trainer", "lightgbm")
    """
    
    _registries: Dict[str, Dict[str, Type]] = {
        "trainer": {},
        "evaluator": {},
        "importance_extractor": {},
        "param_space": {},
        "tuner": {},
        "data_adapter": {},
        "data_processor": {},
    }
    
    @classmethod
    def register(
        cls,
        component_type: str,
        name: str,
        component_class: Type,
    ) -> None:
        """
        Register component
        
        Args:
            component_type: Component type (trainer, evaluator, importance_extractor, param_space, tuner, data_adapter, data_processor)
            name: Component name (e.g., lightgbm, xgboost)
            component_class: Component class
        """
        if component_type not in cls._registries:
            raise ValueError(f"Unknown component type: {component_type}. "
                           f"Supported: {list(cls._registries.keys())}")
        cls._registries[component_type][name] = component_class
    
    @classmethod
    def register_trainer(cls, name: str) -> Callable[[Type[BaseTrainer]], Type[BaseTrainer]]:
        """Decorator for registering Trainer"""
        def decorator(component_class: Type[BaseTrainer]) -> Type[BaseTrainer]:
            cls.register("trainer", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_evaluator(cls, name: str) -> Callable[[Type[BaseEvaluator]], Type[BaseEvaluator]]:
        """Decorator for registering Evaluator"""
        def decorator(component_class: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
            cls.register("evaluator", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_importance_extractor(cls, name: str) -> Callable[[Type[BaseImportanceExtractor]], Type[BaseImportanceExtractor]]:
        """Decorator for registering ImportanceExtractor"""
        def decorator(component_class: Type[BaseImportanceExtractor]) -> Type[BaseImportanceExtractor]:
            cls.register("importance_extractor", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_param_space(cls, name: str) -> Callable[[Type[BaseParamSpace]], Type[BaseParamSpace]]:
        """Decorator for registering ParamSpace"""
        def decorator(component_class: Type[BaseParamSpace]) -> Type[BaseParamSpace]:
            cls.register("param_space", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_tuner(cls, name: str) -> Callable[[Type[BaseTuner]], Type[BaseTuner]]:
        """Decorator for registering Tuner"""
        def decorator(component_class: Type[BaseTuner]) -> Type[BaseTuner]:
            cls.register("tuner", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_data_adapter(cls, name: str) -> Callable[[Type[BaseAdapter]], Type[BaseAdapter]]:
        """Decorator for registering DataAdapter"""
        def decorator(component_class: Type[BaseAdapter]) -> Type[BaseAdapter]:
            cls.register("data_adapter", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_data_processor(cls, name: str) -> Callable[[Type[BaseTransformPipeline]], Type[BaseTransformPipeline]]:
        """Decorator for registering DataProcessor"""
        def decorator(component_class: Type[BaseTransformPipeline]) -> Type[BaseTransformPipeline]:
            cls.register("data_processor", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def get(cls, component_type: str, name: str) -> Type:
        """
        Get registered component class
        
        Args:
            component_type: Component type
            name: Component name
            
        Returns:
            Component class
        """
        if component_type not in cls._registries:
            raise ValueError(f"Unknown component type: {component_type}")
        
        registry = cls._registries[component_type]
        if name not in registry:
            available = list(registry.keys())
            raise ValueError(f"Unknown {component_type}: {name}. Available: {available}")
        
        return registry[name]
    
    @classmethod
    def create(cls, component_type: str, name: str, **kwargs) -> Any:
        """
        Create component instance
        
        Args:
            component_type: Component type
            name: Component name
            **kwargs: Component initialization parameters
            
        Returns:
            Component instance
        """
        component_class = cls.get(component_type, name)
        return component_class(**kwargs)
    
    @classmethod
    def list_registered(cls, component_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List registered components
        
        Args:
            component_type: Optional, specify component type
            
        Returns:
            {component_type: [names]}
        """
        if component_type:
            if component_type not in cls._registries:
                raise ValueError(f"Unknown component type: {component_type}")
            return {component_type: list(cls._registries[component_type].keys())}
        return {k: list(v.keys()) for k, v in cls._registries.items()}
    
    @classmethod
    def create_trainer(cls, name: str = "lightgbm", **kwargs) -> BaseTrainer:
        """Create Trainer"""
        return cls.create("trainer", name, **kwargs)
    
    @classmethod
    def create_evaluator(cls, name: str = "lightgbm", **kwargs) -> BaseEvaluator:
        """Create Evaluator"""
        return cls.create("evaluator", name, **kwargs)
    
    @classmethod
    def create_importance_extractor(cls, name: str = "lightgbm", **kwargs) -> BaseImportanceExtractor:
        """Create ImportanceExtractor"""
        return cls.create("importance_extractor", name, **kwargs)
    
    @classmethod
    def create_param_space(cls, name: str = "lightgbm", **kwargs) -> BaseParamSpace:
        """Create ParamSpace"""
        return cls.create("param_space", name, **kwargs)
    
    @classmethod
    def create_tuner(
        cls,
        name: str = "lightgbm",
        param_space: Optional[BaseParamSpace] = None,
        base_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseTuner:
        """Create Tuner"""
        if param_space is None:
            param_space = cls.create_param_space(name)
        if base_params is None:
            base_params = {}
        return cls.create("tuner", name, param_space=param_space, base_params=base_params, **kwargs)
    
    @classmethod
    def create_data_adapter(cls, name: str = "lightgbm", **kwargs) -> BaseAdapter:
        """Create DataAdapter"""
        return cls.create("data_adapter", name, **kwargs)
    
    @classmethod
    def create_data_processor(cls, name: str = "default", **kwargs) -> BaseTransformPipeline:
        """Create DataProcessor"""
        return cls.create("data_processor", name, **kwargs)


def _register_default_components():
    """Register default components"""
    from .lgb import (
        LightGBMAdapter,
        LightGBMEvaluator,
        LightGBMImportanceExtractor,
        LightGBMParamSpace,
        LightGBMTrainer,
        LightGBMTuner,
    )
    from .base import BaseTransformPipeline
    
    ComponentRegistry.register("trainer", "lightgbm", LightGBMTrainer)
    ComponentRegistry.register("evaluator", "lightgbm", LightGBMEvaluator)
    ComponentRegistry.register("importance_extractor", "lightgbm", LightGBMImportanceExtractor)
    ComponentRegistry.register("param_space", "lightgbm", LightGBMParamSpace)
    ComponentRegistry.register("tuner", "lightgbm", LightGBMTuner)
    ComponentRegistry.register("data_adapter", "lightgbm", LightGBMAdapter)
    ComponentRegistry.register("data_processor", "default", BaseTransformPipeline)


# Register default components when module loads
_DEFAULT_COMPONENTS_REGISTERED = False


def register_default_components(force: bool = False) -> None:
    """Register built-in components (idempotent by default)."""
    global _DEFAULT_COMPONENTS_REGISTERED
    if _DEFAULT_COMPONENTS_REGISTERED and not force:
        return
    _register_default_components()
    _DEFAULT_COMPONENTS_REGISTERED = True


# No import-time side effects on the main path.

