"""
Legacy LightGBM evaluator entrypoint.

Main path uses ``gem.method.common.RegressionEvaluator`` directly.
"""

from ..common import RegressionEvaluator


class LightGBMEvaluator(RegressionEvaluator):
    pass
