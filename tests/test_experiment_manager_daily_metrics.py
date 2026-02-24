import unittest

import polars as pl

from gem.experiment.experiment_manager import ExperimentManager


class TestExperimentManagerDailyMetrics(unittest.TestCase):
    def test_aggregate_daily_metric_series_filters_invalid_rows(self) -> None:
        rows = [
            {"split_id": 0, "mode": "test", "metric": "daily_top_ret", "date": 20240101, "value": 1.0},
            {"split_id": 1, "mode": "test", "metric": "daily_top_ret", "date": 20240101, "value": 3.0},
            {"split_id": 0, "mode": "test", "metric": "daily_top_ret", "date": 202401, "value": 10.0},
            {"split_id": 0, "mode": "test", "metric": "daily_top_ret", "date": 20240102, "value": float("inf")},
        ]
        daily = ExperimentManager._aggregate_daily_metric_series(rows)
        self.assertEqual(
            daily.columns,
            ["date", "mode", "metric", "value", "n_split", "is_derived", "source_metric"],
        )
        self.assertEqual(daily.height, 1)
        self.assertEqual(int(daily["date"][0]), 20240101)
        self.assertAlmostEqual(float(daily["value"][0]), 2.0, places=8)
        self.assertEqual(int(daily["n_split"][0]), 2)
        self.assertFalse(bool(daily["is_derived"][0]))
        self.assertIsNone(daily["source_metric"][0])

    def test_append_derived_daily_metrics_values(self) -> None:
        rows = [
            {"split_id": 0, "mode": "test", "metric": "daily_top_ret", "date": 20240101, "value": 1.0},
            {"split_id": 1, "mode": "test", "metric": "daily_top_ret", "date": 20240101, "value": 1.0},
            {"split_id": 0, "mode": "test", "metric": "daily_top_ret", "date": 20240102, "value": 3.0},
            {"split_id": 1, "mode": "test", "metric": "daily_top_ret", "date": 20240102, "value": 3.0},
            {"split_id": 0, "mode": "test", "metric": "daily_benchmark_top_ret", "date": 20240101, "value": 0.5},
            {"split_id": 1, "mode": "test", "metric": "daily_benchmark_top_ret", "date": 20240101, "value": 0.5},
            {"split_id": 0, "mode": "test", "metric": "daily_benchmark_top_ret", "date": 20240102, "value": 1.5},
            {"split_id": 1, "mode": "test", "metric": "daily_benchmark_top_ret", "date": 20240102, "value": 1.5},
            {"split_id": 0, "mode": "test", "metric": "daily_ic", "date": 20240101, "value": 0.2},
            {"split_id": 1, "mode": "test", "metric": "daily_ic", "date": 20240101, "value": 0.2},
            {"split_id": 0, "mode": "test", "metric": "daily_ic", "date": 20240102, "value": 0.4},
            {"split_id": 1, "mode": "test", "metric": "daily_ic", "date": 20240102, "value": 0.4},
        ]
        base = ExperimentManager._aggregate_daily_metric_series(rows)
        derived = ExperimentManager._append_derived_daily_metrics(base)

        top_ret_std = (
            derived.filter(
                (pl.col("mode") == "test") & (pl.col("metric") == "daily_top_ret_std")
            )
            .sort("date")
            ["value"]
            .to_list()
        )
        self.assertEqual(len(top_ret_std), 2)
        self.assertAlmostEqual(float(top_ret_std[0]), 0.0, places=8)
        self.assertAlmostEqual(float(top_ret_std[1]), 1.0, places=8)

        rel_improve = (
            derived.filter(
                (pl.col("mode") == "test")
                & (pl.col("metric") == "daily_top_ret_relative_improve_pct")
            )
            .sort("date")
            ["value"]
            .to_list()
        )
        self.assertEqual(len(rel_improve), 2)
        self.assertAlmostEqual(float(rel_improve[0]), 100.0, places=8)
        self.assertAlmostEqual(float(rel_improve[1]), 100.0, places=8)

        icir = (
            derived.filter(
                (pl.col("mode") == "test") & (pl.col("metric") == "daily_icir_expanding")
            )
            .sort("date")
            ["value"]
            .to_list()
        )
        self.assertEqual(len(icir), 2)
        self.assertIsNone(icir[0])
        self.assertIsNone(icir[1])

        source_metric = (
            derived.filter(pl.col("metric") == "daily_top_ret_std")
            .select("source_metric")
            .to_series()
            .to_list()
        )
        self.assertTrue(all(item == "daily_top_ret" for item in source_metric))

    def test_relative_improve_uses_denom_floor(self) -> None:
        rows = [
            {"split_id": 0, "mode": "test", "metric": "daily_top_ret", "date": 20240101, "value": 1.0},
            {"split_id": 0, "mode": "test", "metric": "daily_benchmark_top_ret", "date": 20240101, "value": 0.0003},
        ]
        base = ExperimentManager._aggregate_daily_metric_series(rows)
        derived = ExperimentManager._append_derived_daily_metrics(base)

        rel_improve = (
            derived.filter(
                (pl.col("mode") == "test")
                & (pl.col("metric") == "daily_top_ret_relative_improve_pct")
            )
            .sort("date")
            ["value"]
            .to_list()
        )
        self.assertEqual(len(rel_improve), 1)
        self.assertAlmostEqual(float(rel_improve[0]), 9997.0, places=8)

    def test_icir_min_periods_outputs_null_before_period_20(self) -> None:
        rows = []
        for idx in range(25):
            rows.append(
                {
                    "split_id": 0,
                    "mode": "test",
                    "metric": "daily_ic",
                    "date": 20240101 + idx,
                    "value": 0.01 * (idx + 1),
                }
            )
        base = ExperimentManager._aggregate_daily_metric_series(rows)
        derived = ExperimentManager._append_derived_daily_metrics(base)
        icir = (
            derived.filter(
                (pl.col("mode") == "test") & (pl.col("metric") == "daily_icir_expanding")
            )
            .sort("date")
            ["value"]
            .to_list()
        )
        self.assertEqual(len(icir), 25)
        self.assertTrue(all(value is None for value in icir[:19]))
        self.assertIsNotNone(icir[19])


if __name__ == "__main__":
    unittest.main()
