"""
GBDT unified module - all components for LightGBM, XGBoost, CatBoost.

Contains:
- Trainers: LightGBMTrainer, XGBoostTrainer, CatBoostTrainer
- Adapters: LightGBMAdapter, XGBoostAdapter, CatBoostAdapter
- Search spaces: LightGBMSpace, XGBoostSpace, CatBoostSpace
- Importance: GBDTImportanceExtractor
"""

from .search_space import (
    LightGBMSpace,
    XGBoostSpace,
    CatBoostSpace,
)
from .importance import GBDTImportanceExtractor
from .adapters import (
    LightGBMAdapter,
    XGBoostAdapter,
    CatBoostAdapter,
)
from .trainers import (
    LightGBMTrainer,
    XGBoostTrainer,
    CatBoostTrainer,
)

__all__ = [
    # Search spaces
    "LightGBMSpace",
    "XGBoostSpace",
    "CatBoostSpace",
    # Importance
    "GBDTImportanceExtractor",
    # Adapters
    "LightGBMAdapter",
    "XGBoostAdapter",
    "CatBoostAdapter",
    # Trainers
    "LightGBMTrainer",
    "XGBoostTrainer",
    "CatBoostTrainer",
]
