"""Hash-related helper utilities."""


import hashlib
import json
from typing import Sequence


def hash_feature_names(feature_names: Sequence[str]) -> str:
    """
    Build a stable short hash for feature-name lists.

    We use a compact JSON serialization so string formatting differences
    (e.g. spaces/quotes from ``str(list)``) do not change the hash.
    """
    payload = json.dumps(
        [str(name) for name in feature_names],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
