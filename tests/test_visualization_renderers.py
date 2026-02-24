import unittest
from pathlib import Path
import shutil
import uuid

import polars as pl

from gem.experiment.experiment_dataclasses import (
    ImportanceVizConfig,
    MetricsVizConfig,
    VisualizationRenderConfig,
)
from gem.visualization.importance_renderer import render_importance_artifacts
from gem.visualization.metrics_renderer import render_metrics_artifacts


class TestVisualizationRenderers(unittest.TestCase):
    def test_importance_renderer_outputs_files(self) -> None:
        df = pl.DataFrame(
            {
                "split_id": [0, 0, 1, 1],
                "test_date_start": [20240101, 20240101, 20240102, 20240102],
                "test_date_end": [20240101, 20240101, 20240102, 20240102],
                "x_label": ["20240101", "20240101", "20240102", "20240102"],
                "feature_idx": [0, 1, 0, 1],
                "feature_name": ["f0", "f1", "f0", "f1"],
                "importance": [0.1, 0.9, 0.2, 0.8],
            }
        )
        output_dir = Path("outputs") / f"test_tmp_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = ImportanceVizConfig(animation=False)
            render_cfg = VisualizationRenderConfig(show=False, interval_ms=200, dpi=90)
            artifacts = render_importance_artifacts(df, output_dir, cfg, render_cfg)
            saved = [item for item in artifacts if item.status == "saved"]
            self.assertTrue(any(item.name == "importance_data" for item in saved))
            self.assertTrue(any(item.name == "importance_heatmap" for item in saved))
            self.assertTrue(any(item.name == "importance_distribution" for item in saved))
            for item in saved:
                assert item.path is not None
                self.assertTrue(item.path.exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_metrics_renderer_outputs_files(self) -> None:
        df = pl.DataFrame(
            {
                "date": [20240101, 20240102, 20240101, 20240102],
                "mode": ["train", "train", "test", "test"],
                "metric": ["daily_ic", "daily_ic", "daily_ic", "daily_ic"],
                "value": [0.1, 0.2, 0.05, 0.08],
                "n_split": [2, 2, 2, 2],
                "is_derived": [False, False, False, False],
                "source_metric": [None, None, None, None],
            }
        ).with_columns(
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias("date_dt")
        )
        output_dir = Path("outputs") / f"test_tmp_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = MetricsVizConfig(metric_names=["daily_ic"])
            render_cfg = VisualizationRenderConfig(show=False, dpi=90)
            artifacts = render_metrics_artifacts(df, output_dir, cfg, render_cfg)
            saved = [item for item in artifacts if item.status == "saved"]
            self.assertTrue(any(item.name == "metrics_data" for item in saved))
            self.assertTrue(any(item.name == "metrics_overview" for item in saved))
            self.assertTrue(any(item.name == "metrics_distribution" for item in saved))
            self.assertTrue(any(item.name.startswith("metric_daily_ic") for item in saved))
            for item in saved:
                assert item.path is not None
                self.assertTrue(item.path.exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
