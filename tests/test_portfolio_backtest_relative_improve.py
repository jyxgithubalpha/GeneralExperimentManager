import unittest

import numpy as np
import polars as pl

from gem.data.data_dataclasses import SplitView
from gem.method.common.portfolio_backtest import (
    PortfolioBacktestCalculator,
    PortfolioBacktestConfig,
)


class TestPortfolioBacktestRelativeImprove(unittest.TestCase):
    def test_relative_improve_uses_denom_floor(self) -> None:
        calculator = PortfolioBacktestCalculator(
            PortfolioBacktestConfig(
                top_k=1,
                money=1.0,
                min_trade_money=0.0,
                ret_scale=1.0,
                use_benchmark_ensemble=False,
                benchmark_col_candidates=("score_value",),
            )
        )

        view = SplitView(
            indices=np.array([0, 1], dtype=np.int64),
            X=np.zeros((2, 1), dtype=np.float64),
            y=np.zeros((2, 1), dtype=np.float64),
            keys=pl.DataFrame(
                {
                    "date": [20240101, 20240101],
                    "code": ["A", "B"],
                }
            ),
            feature_names=["f0"],
            label_names=["y"],
            extra=pl.DataFrame(
                {
                    "ret_value": [0.2, 0.0003],
                    "liquidity_value": [1.0, 1.0],
                    "score_value": [0.0, 1.0],
                }
            ),
        )

        metrics, _ = calculator.compute(pred=np.array([1.0, 0.0]), view=view)
        expected = (0.2 - 0.0003) / 0.01 * 100.0
        self.assertAlmostEqual(
            float(metrics[PortfolioBacktestCalculator.TOP_RET_RELATIVE_IMPROVE_PCT]),
            expected,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()

