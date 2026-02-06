"""
BayesianTrainer - 贝叶斯回归模型训练器
"""
from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ..training_dataclasses import FitResult, TrainConfig

from ..base import BaseTrainer
from ...data.data_adapter import NumpyAdapter


class BayesianTrainer(BaseTrainer):
    """
    贝叶斯回归模型训练器
    
    支持:
    - BayesianRidge: 贝叶斯岭回归
    - ARDRegression: 自动相关性确定回归
    - GaussianProcessRegressor: 高斯过程回归 (小数据集)
    """
    
    def __init__(
        self,
        adapter: Optional[NumpyAdapter] = None,
        model_type: str = "bayesian_ridge",
    ):
        self.adapter = adapter or NumpyAdapter()
        self.model_type = model_type
    
    def _create_model(self, params: dict):
        """创建贝叶斯模型"""
        model_type = params.get("model_type", self.model_type)
        
        if model_type == "bayesian_ridge":
            from sklearn.linear_model import BayesianRidge
            return BayesianRidge(
                alpha_1=params.get("alpha_1", 1e-6),
                alpha_2=params.get("alpha_2", 1e-6),
                lambda_1=params.get("lambda_1", 1e-6),
                lambda_2=params.get("lambda_2", 1e-6),
                n_iter=params.get("n_iter", 300),
                tol=params.get("tol", 1e-3),
                compute_score=True,
            )
        elif model_type == "ard":
            from sklearn.linear_model import ARDRegression
            return ARDRegression(
                alpha_1=params.get("alpha_1", 1e-6),
                alpha_2=params.get("alpha_2", 1e-6),
                lambda_1=params.get("lambda_1", 1e-6),
                lambda_2=params.get("lambda_2", 1e-6),
                n_iter=params.get("n_iter", 300),
                tol=params.get("tol", 1e-3),
                compute_score=True,
            )
        elif model_type == "gaussian_process":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
            
            length_scale = params.get("length_scale", 1.0)
            noise_level = params.get("noise_level", 0.1)
            
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
            return GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=params.get("n_restarts_optimizer", 5),
                normalize_y=True,
                random_state=params.get("seed", 42),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        mode: str = "full",
    ) -> "FitResult":
        """
        训练贝叶斯模型
        
        Args:
            views: 处理后的视图
            config: 训练配置
            mode: "full" 或 "tune"
            
        Returns:
            FitResult 实例
        """
        from ..training_dataclasses import FitResult
        
        start_time = time.time()
        
        # 设置随机种子
        np.random.seed(config.seed)
        
        # 获取参数
        params = dict(config.params)
        params["seed"] = config.seed
        
        # 构建数据
        X_train, y_train = self.adapter.to_dataset(views.train)
        X_val, y_val = self.adapter.to_dataset(views.val)
        
        # 对于高斯过程，如果数据太大，进行子采样
        model_type = params.get("model_type", self.model_type)
        if model_type == "gaussian_process" and len(X_train) > 5000:
            indices = np.random.choice(len(X_train), 5000, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        # 创建并训练模型
        model = self._create_model(params)
        model.fit(X_train, y_train)
        
        # 构建 evals_result
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_mse = np.mean((train_pred - y_train) ** 2)
        val_mse = np.mean((val_pred - y_val) ** 2)
        
        evals_result = {
            "train": {"mse": [train_mse]},
            "val": {"mse": [val_mse]},
        }
        
        # 添加模型特定的分数
        if hasattr(model, "scores_"):
            evals_result["train"]["score"] = list(model.scores_)
        
        train_time = time.time() - start_time
        
        verbose = mode == "full"
        if verbose:
            print(f"Bayesian Model trained in {train_time:.2f}s")
            print(f"Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
        
        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=1,
            params=params,
            seed=config.seed,
            train_time=train_time,
        )
