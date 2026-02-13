"""
PyTorch trainer for tabular regression models.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ...data.data_dataclasses import ProcessedViews
from ..base import BaseTrainer
from ..method_dataclasses import FitResult, TrainConfig
from .torch_models import MLPRegressor, FTTransformerRegressor


class TorchModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            preds = self.model(x).detach().cpu().numpy()
        return preds


class TorchTabularTrainer(BaseTrainer):
    def __init__(
        self,
        model_type: str = "mlp",
        use_gpu: Optional[bool] = None,
        epochs: int = 50,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.model_params = model_params or {}

    def fit(
        self,
        views: "ProcessedViews",
        config: TrainConfig,
        mode: str = "full",
        sample_weights: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        start_time = time.time()
        device = self._resolve_device(config)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        X_train = views.train.X.astype(np.float32)
        y_train = views.train.y.ravel().astype(np.float32)
        X_val = views.val.X.astype(np.float32)
        y_val = views.val.y.ravel().astype(np.float32)

        train_weights = None
        if sample_weights and sample_weights.get("train") is not None:
            train_weights = sample_weights.get("train").astype(np.float32)

        model = self._build_model(input_dim=X_train.shape[1], config=config)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_loader = self._build_loader(X_train, y_train, train_weights, self.batch_size, shuffle=True)
        val_loader = self._build_loader(X_val, y_val, None, self.batch_size, shuffle=False)

        best_state = None
        best_val = float("inf")
        best_epoch = 0
        val_history = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            for batch in train_loader:
                xb, yb, wb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                if wb is not None:
                    wb = wb.to(device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = self._weighted_mse(preds, yb, wb)
                loss.backward()
                optimizer.step()

            val_loss = self._evaluate_mse(model, val_loader, device)
            val_history.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        wrapper = TorchModelWrapper(model, device)
        evals_result = {"val": {"mse": val_history}}

        return FitResult(
            model=wrapper,
            evals_result=evals_result,
            best_iteration=max(1, best_epoch),
            params=dict(config.params),
            seed=config.seed,
            train_time=time.time() - start_time,
        )

    def _build_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray],
        batch_size: int,
        shuffle: bool,
    ):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        xb = torch.from_numpy(X)
        yb = torch.from_numpy(y)
        if weights is not None:
            wb = torch.from_numpy(weights)
        else:
            wb = torch.ones(xb.shape[0], dtype=torch.float32)

        dataset = TensorDataset(xb, yb, wb)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    @staticmethod
    def _weighted_mse(preds, y_true, weights):
        import torch

        diff = preds - y_true
        if weights is None or weights.numel() == 0:
            return torch.mean(diff ** 2)
        return torch.mean(weights * (diff ** 2))

    @staticmethod
    def _evaluate_mse(model, loader, device):
        import torch

        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                losses.append(torch.mean((preds - yb) ** 2).item())
        if not losses:
            return 0.0
        return float(np.mean(losses))

    def _build_model(self, input_dim: int, config: TrainConfig):
        params = dict(self.model_params)
        params.update(config.params or {})

        model_type = str(params.pop("model_type", self.model_type)).lower()
        if model_type == "mlp":
            return MLPRegressor(
                input_dim=input_dim,
                hidden_sizes=params.get("hidden_sizes"),
                dropout=float(params.get("dropout", 0.0)),
                activation=str(params.get("activation", "relu")),
            )
        if model_type in {"ft_transformer", "ft"}:
            return FTTransformerRegressor(
                input_dim=input_dim,
                d_token=int(params.get("d_token", 64)),
                n_heads=int(params.get("n_heads", 4)),
                n_layers=int(params.get("n_layers", 3)),
                dropout=float(params.get("dropout", 0.1)),
                activation=str(params.get("activation", "relu")),
                use_cls_token=bool(params.get("use_cls_token", True)),
            )
        raise ValueError(f"Unknown model_type '{model_type}'.")

    def _resolve_device(self, config: TrainConfig):
        import torch

        use_gpu = self.use_gpu
        if use_gpu is None:
            use_gpu = bool(config.params.get("use_gpu", False))
        if use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
