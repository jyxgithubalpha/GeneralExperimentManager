from __future__ import annotations

from typing import List, Optional, Sequence, Union


def to_clean_list(items: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Normalize a scalar/list config field into a cleaned string list."""
    if items is None:
        return []
    if isinstance(items, str):
        return [items.strip('"\'')]
    return [str(item).strip('"\'') for item in items]


def remove_quotes_from_list(items: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Backward-compatible alias used by existing config plumbing."""
    return to_clean_list(items)
