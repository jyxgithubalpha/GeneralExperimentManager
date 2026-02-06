"""
Component Registry - 组件注册表

提供统一的接口来注册和创建不同机器学习框架的组件实例
支持动态注册新的组件实现
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from .base import (
    BaseEvaluator,
    BaseImportanceExtractor,
    BaseParamSpace,
    BaseTrainer,
    BaseTuner,
)


class ComponentRegistry:
    """
    组件注册表
    
    使用装饰器或显式注册方法来注册组件
    
    Example:
        # 使用装饰器注册
        @ComponentRegistry.register_trainer("lightgbm")
        class LightGBMTrainer(BaseTrainer):
            ...
        
        # 显式注册
        ComponentRegistry.register("trainer", "xgboost", XGBoostTrainer)
        
        # 创建组件
        trainer = ComponentRegistry.create("trainer", "lightgbm")
    """
    
    _registries: Dict[str, Dict[str, Type]] = {
        "trainer": {},
        "evaluator": {},
        "importance_extractor": {},
        "param_space": {},
        "tuner": {},
    }
    
    @classmethod
    def register(
        cls,
        component_type: str,
        name: str,
        component_class: Type,
    ) -> None:
        """
        注册组件
        
        Args:
            component_type: 组件类型 (trainer, evaluator, importance_extractor, param_space, tuner)
            name: 组件名称 (如 lightgbm, xgboost)
            component_class: 组件类
        """
        if component_type not in cls._registries:
            raise ValueError(f"Unknown component type: {component_type}. "
                           f"Supported: {list(cls._registries.keys())}")
        cls._registries[component_type][name] = component_class
    
    @classmethod
    def register_trainer(cls, name: str) -> Callable[[Type[BaseTrainer]], Type[BaseTrainer]]:
        """注册 Trainer 的装饰器"""
        def decorator(component_class: Type[BaseTrainer]) -> Type[BaseTrainer]:
            cls.register("trainer", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_evaluator(cls, name: str) -> Callable[[Type[BaseEvaluator]], Type[BaseEvaluator]]:
        """注册 Evaluator 的装饰器"""
        def decorator(component_class: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
            cls.register("evaluator", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_importance_extractor(cls, name: str) -> Callable[[Type[BaseImportanceExtractor]], Type[BaseImportanceExtractor]]:
        """注册 ImportanceExtractor 的装饰器"""
        def decorator(component_class: Type[BaseImportanceExtractor]) -> Type[BaseImportanceExtractor]:
            cls.register("importance_extractor", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_param_space(cls, name: str) -> Callable[[Type[BaseParamSpace]], Type[BaseParamSpace]]:
        """注册 ParamSpace 的装饰器"""
        def decorator(component_class: Type[BaseParamSpace]) -> Type[BaseParamSpace]:
            cls.register("param_space", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def register_tuner(cls, name: str) -> Callable[[Type[BaseTuner]], Type[BaseTuner]]:
        """注册 Tuner 的装饰器"""
        def decorator(component_class: Type[BaseTuner]) -> Type[BaseTuner]:
            cls.register("tuner", name, component_class)
            return component_class
        return decorator
    
    @classmethod
    def get(cls, component_type: str, name: str) -> Type:
        """
        获取已注册的组件类
        
        Args:
            component_type: 组件类型
            name: 组件名称
            
        Returns:
            组件类
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
        创建组件实例
        
        Args:
            component_type: 组件类型
            name: 组件名称
            **kwargs: 组件初始化参数
            
        Returns:
            组件实例
        """
        component_class = cls.get(component_type, name)
        return component_class(**kwargs)
    
    @classmethod
    def list_registered(cls, component_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        列出已注册的组件
        
        Args:
            component_type: 可选，指定组件类型
            
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
        """创建 Trainer"""
        return cls.create("trainer", name, **kwargs)
    
    @classmethod
    def create_evaluator(cls, name: str = "lightgbm", **kwargs) -> BaseEvaluator:
        """创建 Evaluator"""
        return cls.create("evaluator", name, **kwargs)
    
    @classmethod
    def create_importance_extractor(cls, name: str = "lightgbm", **kwargs) -> BaseImportanceExtractor:
        """创建 ImportanceExtractor"""
        return cls.create("importance_extractor", name, **kwargs)
    
    @classmethod
    def create_param_space(cls, name: str = "lightgbm", **kwargs) -> BaseParamSpace:
        """创建 ParamSpace"""
        return cls.create("param_space", name, **kwargs)
    
    @classmethod
    def create_tuner(
        cls,
        name: str = "lightgbm",
        param_space: Optional[BaseParamSpace] = None,
        base_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseTuner:
        """创建 Tuner"""
        if param_space is None:
            param_space = cls.create_param_space(name)
        if base_params is None:
            base_params = {}
        return cls.create("tuner", name, param_space=param_space, base_params=base_params, **kwargs)


def _register_default_components():
    """注册默认的 LightGBM 组件"""
    from .lgb import (
        LightGBMEvaluator,
        LightGBMImportanceExtractor,
        LightGBMParamSpace,
        LightGBMTrainer,
        LightGBMTuner,
    )
    
    ComponentRegistry.register("trainer", "lightgbm", LightGBMTrainer)
    ComponentRegistry.register("evaluator", "lightgbm", LightGBMEvaluator)
    ComponentRegistry.register("importance_extractor", "lightgbm", LightGBMImportanceExtractor)
    ComponentRegistry.register("param_space", "lightgbm", LightGBMParamSpace)
    ComponentRegistry.register("tuner", "lightgbm", LightGBMTuner)


# 模块加载时注册默认组件
_register_default_components()
