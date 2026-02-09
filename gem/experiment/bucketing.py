"""
Bucketing - Bucket 管理和辅助函数
"""

from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.data_dataclasses import SplitSpec
    from .state_dataclasses import StatePolicyMode


def quarter_bucket_fn(splitspec: "SplitSpec") -> str:
    """按季度分组的 bucket 函数"""
    test_start = splitspec.test_date_list[0] if splitspec.test_date_list else 0
    year = test_start // 10000
    month = (test_start % 10000) // 100
    quarter = (month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def month_bucket_fn(splitspec: "SplitSpec") -> str:
    """按月分组的 bucket 函数"""
    test_start = splitspec.test_date_list[0] if splitspec.test_date_list else 0
    year = test_start // 10000
    month = (test_start % 10000) // 100
    return f"{year}M{month:02d}"


class BucketManager:
    """
    Bucket 管理器
    
    负责:
    - 将 splits 分组到 buckets
    - 管理 bucket 间的执行顺序
    """
    
    def __init__(
        self,
        bucket_fn: Optional[Callable[["SplitSpec"], str]] = None,
    ):
        self.bucket_fn = bucket_fn or quarter_bucket_fn
    
    def group_splits(
        self,
        splitspecs: List["SplitSpec"],
    ) -> Dict[str, List["SplitSpec"]]:
        """
        将 splits 分组到 buckets
        """
        buckets: Dict[str, List["SplitSpec"]] = {}
        
        for spec in splitspecs:
            key = self.bucket_fn(spec)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(spec)
        
        return buckets
    
    def get_bucket_order(
        self,
        buckets: Dict[str, List["SplitSpec"]],
    ) -> List[str]:
        """
        获取 bucket 执行顺序 (按时间排序)
        """
        def get_min_test_start(key: str) -> int:
            return min(s.test_date_list[0] if s.test_date_list else 0 for s in buckets[key])
        
        return sorted(buckets.keys(), key=get_min_test_start)
    
    def create_execution_plan(
        self,
        splitspecs: List["SplitSpec"],
        mode: "StatePolicyMode",
    ) -> List[List["SplitSpec"]]:
        """
        创建执行计划
        """
        from .state_dataclasses import StatePolicyMode
        
        if mode == StatePolicyMode.NONE:
            # 全并行
            return [splitspecs]
        elif mode == StatePolicyMode.PER_SPLIT:
            # 严格串行
            sorted_specs = sorted(splitspecs, key=lambda s: s.test_date_list[0] if s.test_date_list else 0)
            return [[s] for s in sorted_specs]
        elif mode == StatePolicyMode.BUCKET:
            # Bucket 内并行，Bucket 间串行
            buckets = self.group_splits(splitspecs)
            order = self.get_bucket_order(buckets)
            return [buckets[key] for key in order]
        else:
            return [splitspecs]
