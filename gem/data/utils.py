
from typing import List, Optional, Sequence, Union


def to_clean_list(items: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Normalize a scalar/list config field into a cleaned string list."""
    if items is None:
        return []
    if isinstance(items, str):
        return [items.strip('"\'')]
    return [str(item).strip('"\'') for item in items]
