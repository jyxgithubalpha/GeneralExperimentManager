import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import shutil
import uuid

import numpy as np
import polars as pl

from gem.data.data_dataclasses import SplitSpec
from gem.experiment.experiment_dataclasses import (
    ImportanceVizConfig,
    MetricsVizConfig,
    SplitResult,
    VisualizationConfig,
    VisualizationRenderConfig,
)
from gem.visualization.pipeline import run_visualization_pipeline


class TestVisualizationPipeline(unittest.TestCase):
    def test_pipeline_generates_artifacts_and_manifest(self) -> None:
        output_dir = Path("outputs") / f"test_tmp_{uuid.uuid4().hex}"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            metric_csv = output_dir / "daily_metric_series.csv"
            pl.DataFrame(
                {
                    "date": [20240101, 20240102, 20240101, 20240102],
                    "mode": ["test", "test", "train", "train"],
                    "metric": ["daily_ic", "daily_ic", "daily_ic", "daily_ic"],
                    "value": [0.1, 0.12, 0.2, 0.25],
                    "n_split": [1, 1, 1, 1],
                    "is_derived": [False, False, False, False],
                    "source_metric": [None, None, None, None],
                }
            ).write_csv(metric_csv)

            results = {
                0: SplitResult(split_id=0, importance_vector=np.array([0.2, 0.8], dtype=np.float64))
            }
            manager = SimpleNamespace(
                experiment_config=SimpleNamespace(
                    visualization=VisualizationConfig(
                        enabled=True,
                        output_subdir="visualization",
                        render=VisualizationRenderConfig(show=False, interval_ms=200, dpi=90),
                        importance=ImportanceVizConfig(animation=False),
                        metrics=MetricsVizConfig(metric_names=["daily_ic"], per_metric=False),
                    )
                ),
                splitspec_list=[
                    SplitSpec(
                        split_id=0,
                        train_date_list=[20230101],
                        val_date_list=[20230102],
                        test_date_list=[20230103],
                    )
                ],
                feature_names=["f0", "f1"],
            )

            saved_paths = run_visualization_pipeline(manager, results, output_dir)
            manifest_path = output_dir / "visualization" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(any(path.name == "manifest.json" for path in saved_paths))

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("artifacts", payload)
            names = {item["name"] for item in payload["artifacts"]}
            self.assertIn("importance_data", names)
            self.assertIn("metrics_data", names)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
