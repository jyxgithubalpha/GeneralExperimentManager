"""
工具函数 - 数据适配和默认配置
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..data.data_dataclasses import SplitSpec
from ..data.data_dataclasses import DataBundle, SplitData
from .hooks import (
    FeatureImportanceHook,
    HookManager,
    SaveModelHook,
    SavePredictionsHook,
    TrainLogHook,
)
from ..data.data_processor import StandardizeTransform, WinsorizeTransform


def convert_legacy_split_data(legacy_data: Dict[str, Dict[str, Any]], 
                               split_spec: SplitSpec,
                               feature_names: Optional[List[str]] = None) -> SplitData:
    """将旧格式数据转换为新格式SplitData"""
    bundles = {}
    
    for mode in ["train", "val", "test"]:
        data = legacy_data[mode]
        X = data["X"]
        y = data["y"].ravel() if data["y"].ndim > 1 else data["y"]
        
        # 构建meta DataFrame
        meta = data["extra_df"].copy()
        
        # 标准化列名
        col_mapping = {}
        for col in meta.columns:
            if "liquidity" in col.lower():
                col_mapping[col] = "liquidity"
            elif "ret_raw" in col.lower() or "ret_value" in col.lower():
                col_mapping[col] = "ret"
            elif "score" in col.lower():
                col_mapping[col] = "score"
        
        if col_mapping:
            meta = meta.rename(columns=col_mapping)
        
        bundles[mode] = DataBundle(
            X=X,
            y=y,
            meta=meta,
            feature_names=feature_names
        )
    
    return SplitData(
        train=bundles["train"],
        val=bundles["val"],
        test=bundles["test"],
        split_spec=split_spec
    )


def create_default_hooks(mode: str = "full") -> HookManager:
    """创建默认Hook集合"""
    manager = HookManager()
    
    if mode == "full":
        manager.register(SaveModelHook())
        manager.register(FeatureImportanceHook())
        manager.register(SavePredictionsHook())
        manager.register(TrainLogHook())
    else:
        manager.register(TrainLogHook())
    
    return manager


def create_default_transforms():
    """创建默认Transform列表"""
    return [
        WinsorizeTransform(lower=0.01, upper=0.99),
        StandardizeTransform(eps=1e-8),
    ]


