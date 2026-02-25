
import logging
from typing import Mapping, Optional, Sequence

import numpy as np
import polars as pl

from ...data.data_dataclasses import SplitSpec
from ...experiment.results import SplitResult

log = logging.getLogger(__name__)


def _feature_name(idx: int, feature_names: Optional[Sequence[str]]) -> str:
    if feature_names is not None and idx < len(feature_names):
        return str(feature_names[idx])
    return f"F{idx}"


def _split_meta_map(splitspec_list: Optional[Sequence[SplitSpec]]) -> dict[int, tuple[Optional[int], Optional[int]]]:
    mapping: dict[int, tuple[Optional[int], Optional[int]]] = {}
    if not splitspec_list:
        return mapping
    for spec in splitspec_list:
        start = spec.test_date_list[0] if spec.test_date_list else None
        end = spec.test_date_list[-1] if spec.test_date_list else None
        mapping[spec.split_id] = (start, end)
    return mapping


def build_importance_dataframe(
    results: Mapping[int, SplitResult],
    splitspec_list: Optional[Sequence[SplitSpec]],
    feature_names: Optional[Sequence[str]],
) -> Optional[pl.DataFrame]:
    if not results:
        return None

    split_meta = _split_meta_map(splitspec_list)
    rows: list[dict] = []
    expected_n_features: Optional[int] = None

    for split_id in sorted(results.keys()):
        result = results[split_id]
        if result.skipped or result.failed or result.importance_vector is None:
            continue

        vector = np.asarray(result.importance_vector, dtype=np.float64).ravel()
        if vector.size == 0:
            continue
        if not np.all(np.isfinite(vector)):
            log.warning("Skip split %s importance because non-finite values were found.", split_id)
            continue

        if expected_n_features is None:
            expected_n_features = int(vector.size)
        elif int(vector.size) != expected_n_features:
            log.warning(
                "Skip split %s importance due to length mismatch: expected=%d got=%d",
                split_id,
                expected_n_features,
                int(vector.size),
            )
            continue

        test_start, test_end = split_meta.get(split_id, (None, None))
        x_label = str(test_start) if test_start is not None else f"split_{split_id}"
        for idx, value in enumerate(vector):
            rows.append(
                {
                    "split_id": split_id,
                    "test_date_start": test_start,
                    "test_date_end": test_end,
                    "x_label": x_label,
                    "feature_idx": idx,
                    "feature_name": _feature_name(idx, feature_names),
                    "importance": float(value),
                }
            )

    if not rows:
        return None

    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("split_id").cast(pl.Int64),
            pl.col("test_date_start").cast(pl.Int64, strict=False),
            pl.col("test_date_end").cast(pl.Int64, strict=False),
            pl.col("x_label").cast(pl.Utf8),
            pl.col("feature_idx").cast(pl.Int64),
            pl.col("feature_name").cast(pl.Utf8),
            pl.col("importance").cast(pl.Float64),
        )
        .sort(["split_id", "feature_idx"])
    )
