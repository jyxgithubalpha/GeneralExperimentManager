"""
MLPTrainer - MLP 模型训练器
"""
from __future__ import annotations

import time
from typing import Optional, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...data.data_dataclasses import ProcessedViews
    from ..training_dataclasses import FitResult, TrainConfig

from ..base import BaseTrainer
from ...data.data_adapter import PyTorchAdapter


class MLPTrainer(BaseTrainer):
    """
    MLP 模型训练器
    
    支持:
    - PyTorch MLP
    - 自定义网络结构
    - 早停和学习率调度
    """
    
    def __init__(
        self,
        adapter: Optional[PyTorchAdapter] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        self.adapter = adapter or PyTorchAdapter()
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.batch_norm = batch_norm
    
    def _build_model(self, input_dim: int, params: dict):
        """构建 MLP 模型"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required for MLPTrainer")
        
        hidden_dims = params.get("hidden_dims", self.hidden_dims)
        dropout = params.get("dropout", self.dropout)
        batch_norm = params.get("batch_norm", self.batch_norm)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def fit(
        self,
        views: "ProcessedViews",
        config: "TrainConfig",
        mode: str = "full",
    ) -> "FitResult":
        """
        训练 MLP 模型
        
        Args:
            views: 处理后的视图
            config: 训练配置
            mode: "full" 或 "tune"
            
        Returns:
            FitResult 实例
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("torch is required for MLPTrainer")
        
        from ..training_dataclasses import FitResult
        
        start_time = time.time()
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 获取参数
        params = dict(config.params)
        learning_rate = params.get("learning_rate", 0.001)
        batch_size = params.get("batch_size", 1024)
        
        # 构建 DataLoader
        self.adapter.batch_size = batch_size
        train_loader, val_loader, _ = self.adapter.to_train_val_test(views)
        
        # 构建模型
        input_dim = views.train.X.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(input_dim, params).to(device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练
        epochs = config.num_boost_round
        early_stopping_rounds = config.early_stopping_rounds
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        evals_result = {"train": {"loss": []}, "val": {"loss": []}}
        
        verbose = mode == "full"
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            evals_result["train"]["loss"].append(train_loss)
            
            # 验证阶段
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            evals_result["val"]["loss"].append(val_loss)
            
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % config.verbose_eval == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
            
            if patience_counter >= early_stopping_rounds:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 恢复最佳模型
        model.load_state_dict(best_model_state)
        
        train_time = time.time() - start_time
        
        return FitResult(
            model=model,
            evals_result=evals_result,
            best_iteration=best_epoch + 1,
            params=params,
            seed=config.seed,
            train_time=train_time,
        )
