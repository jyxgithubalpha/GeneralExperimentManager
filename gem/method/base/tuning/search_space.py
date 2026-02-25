"""
BaseSearchSpace - Unified search space for parameter and architecture search.

Supports:
- Optuna sampling
- Ray Tune space conversion
- NNI space conversion
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BaseSearchSpace(ABC):
    """
    Base class for search spaces.
    
    Supports both hyperparameter search (GBDT, sklearn) and 
    architecture search (PyTorch NAS).
    
    Subclasses define parameters as dataclass fields with Tuple ranges
    or List choices.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Auto-generate dict from dataclass fields."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def get_param_names(self) -> List[str]:
        """Get all parameter names."""
        return [f.name for f in fields(self)]
    
    @abstractmethod
    def sample_optuna(self, trial, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sample parameters using Optuna trial.
        
        Args:
            trial: Optuna trial object
            shrunk_space: Optional shrunk search space
            
        Returns:
            Sampled parameter dict
        """
        pass
    
    def to_ray_tune_space(self, shrunk_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert to Ray Tune search space.
        
        Args:
            shrunk_space: Optional shrunk search space
            
        Returns:
            Ray Tune search space dict
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement to_ray_tune_space()"
        )
    
    def to_nni_space(self) -> Dict[str, Any]:
        """
        Convert to NNI search space.
        
        Returns:
            NNI search space dict
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement to_nni_space()"
        )
    
    def get_shrunk_space(
        self,
        best_params: Dict[str, Any],
        shrink_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Shrink search space around best parameters.
        
        Args:
            best_params: Best parameters from previous search
            shrink_ratio: Ratio to shrink the space (0.5 = half the range)
            
        Returns:
            Shrunk search space dict
        """
        original = self.to_dict()
        shrunk = {}
        
        for name, bounds in original.items():
            if name not in best_params:
                shrunk[name] = bounds
                continue
            
            best_val = best_params[name]
            
            if isinstance(bounds, tuple) and len(bounds) == 2:
                low, high = bounds
                if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    range_size = (high - low) * shrink_ratio
                    new_low = max(low, best_val - range_size / 2)
                    new_high = min(high, best_val + range_size / 2)
                    if isinstance(low, int) and isinstance(high, int):
                        shrunk[name] = (int(new_low), int(new_high))
                    else:
                        shrunk[name] = (new_low, new_high)
                else:
                    shrunk[name] = bounds
            else:
                shrunk[name] = bounds
        
        return shrunk
