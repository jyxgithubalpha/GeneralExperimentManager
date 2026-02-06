"""
BaseParamSpace - 参数搜索空间基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseParamSpace(ABC):
    """参数搜索空间基类"""
    
    @abstractmethod
    def sample(self, trial: Any) -> Dict[str, Any]:
        """采样参数"""
        pass
