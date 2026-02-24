import unittest
from pathlib import Path
import shutil
import uuid

import polars as pl

from gem.visualization.metrics_data import load_metrics_data


class TestMetricsData(unittest.TestCase):
    def test_load_metrics_data_reports_missing_columns(self) -> None:
        tmp_dir = Path("outputs") / f"test_tmp_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp_dir / "daily_metric_series.csv"
            pl.DataFrame({"foo": [1]}).write_csv(path)
            df, err = load_metrics_data(path)
            self.assertIsNone(df)
            self.assertIsNotNone(err)
            assert err is not None
            self.assertIn("missing required columns", err)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_load_metrics_data_filters_invalid_rows(self) -> None:
        tmp_dir = Path("outputs") / f"test_tmp_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp_dir / "daily_metric_series.csv"
            pl.DataFrame(
                {
                    "date": [20240101, 202401, 20240102],
                    "mode": ["test", "test", "test"],
                    "metric": ["daily_ic", "daily_ic", "daily_ic"],
                    "value": [0.1, 0.2, float("inf")],
                    "n_split": [2, 2, 2],
                    "is_derived": [False, False, False],
                    "source_metric": [None, None, None],
                }
            ).write_csv(path)

            df, err = load_metrics_data(path)
            self.assertIsNone(err)
            self.assertIsNotNone(df)
            assert df is not None
            self.assertEqual(df.height, 1)
            self.assertEqual(int(df["date"][0]), 20240101)
            self.assertIn("date_dt", df.columns)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
