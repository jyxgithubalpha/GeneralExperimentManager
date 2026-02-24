import unittest

import numpy as np

from gem.data.data_dataclasses import SplitSpec
from gem.experiment.experiment_dataclasses import SplitResult
from gem.visualization.importance_data import build_importance_dataframe


class TestImportanceData(unittest.TestCase):
    def test_build_importance_dataframe_skips_invalid_splits(self) -> None:
        results = {
            0: SplitResult(split_id=0, importance_vector=np.array([0.2, 0.8], dtype=np.float64)),
            1: SplitResult(split_id=1, skipped=True),
            2: SplitResult(split_id=2, importance_vector=np.array([0.1, 0.2, 0.3], dtype=np.float64)),
        }
        splitspec_list = [
            SplitSpec(split_id=0, train_date_list=[20230101], val_date_list=[20230102], test_date_list=[20230103]),
            SplitSpec(split_id=1, train_date_list=[20230111], val_date_list=[20230112], test_date_list=[20230113]),
            SplitSpec(split_id=2, train_date_list=[20230121], val_date_list=[20230122], test_date_list=[20230123]),
        ]
        feature_names = ["alpha", "beta"]

        df = build_importance_dataframe(results, splitspec_list, feature_names)
        self.assertIsNotNone(df)
        assert df is not None

        self.assertEqual(df.height, 2)
        self.assertEqual(df["split_id"].unique().to_list(), [0])
        self.assertEqual(df["x_label"].unique().to_list(), ["20230103"])
        self.assertEqual(df["feature_name"].to_list(), ["alpha", "beta"])

    def test_build_importance_dataframe_returns_none_when_no_valid_data(self) -> None:
        results = {
            0: SplitResult(split_id=0, skipped=True),
            1: SplitResult(split_id=1, failed=True),
        }
        df = build_importance_dataframe(results, splitspec_list=None, feature_names=None)
        self.assertIsNone(df)


if __name__ == "__main__":
    unittest.main()

