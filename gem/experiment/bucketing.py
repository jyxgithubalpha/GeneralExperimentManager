"""
Bucket utilities for execution planning.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..data.data_dataclasses import SplitSpec


def _get_test_start(splitspec: SplitSpec) -> int:
    return splitspec.test_date_list[0] if splitspec.test_date_list else 0


def quarter_bucket_fn(splitspec: SplitSpec) -> str:
    test_start = _get_test_start(splitspec)
    year = test_start // 10000
    month = (test_start % 10000) // 100
    quarter = (month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def month_bucket_fn(splitspec: SplitSpec) -> str:
    test_start = _get_test_start(splitspec)
    year = test_start // 10000
    month = (test_start % 10000) // 100
    return f"{year}M{month:02d}"


class BucketManager:
    def __init__(self, bucket_fn: Optional[Callable[[SplitSpec], str]] = None):
        self.bucket_fn = bucket_fn or quarter_bucket_fn

    def group_splits(self, splitspecs: List[SplitSpec]) -> Dict[str, List[SplitSpec]]:
        buckets: Dict[str, List[SplitSpec]] = {}
        for spec in splitspecs:
            key = self.bucket_fn(spec)
            buckets.setdefault(key, []).append(spec)
        return buckets

    def get_bucket_order(self, buckets: Dict[str, List[SplitSpec]]) -> List[str]:
        return sorted(
            buckets.keys(),
            key=lambda key: min(_get_test_start(spec) for spec in buckets[key]),
        )

    def create_execution_plan(self, splitspecs: List[SplitSpec], mode: str) -> List[List[SplitSpec]]:
        if mode == "none":
            return [list(splitspecs)]

        sorted_specs = sorted(splitspecs, key=_get_test_start)
        if mode == "per_split":
            return [[spec] for spec in sorted_specs]

        if mode == "bucket":
            buckets = self.group_splits(sorted_specs)
            order = self.get_bucket_order(buckets)
            return [buckets[key] for key in order]

        raise ValueError(
            f"Unknown execution mode '{mode}'. Expected one of: none, per_split, bucket."
        )
