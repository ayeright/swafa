from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer, Adam
from pytorch_lightning import LightningModule


class FeedForwardNet(LightningModule):

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None,
                 activation_fn: Optional[nn.Module] = None, optimiser_class: Optimizer = Adam,
                 optimiser_kwargs: Optional[dict] = None, loss_fn: nn.Module = nn.MSELoss(),
                 random_seed: Optional[int] = None):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or []
        self.activation_fn = activation_fn or self._identity_activation_fn
        self.optimiser_class = optimiser_class
        self.optimiser_kwargs = optimiser_kwargs or dict(lr=0.001)
        self.loss_fn = loss_fn

        self.hidden_layers = nn.ModuleList()
        d_in = deepcopy(input_dim)
        for d_out in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(d_in, d_out))
            d_in = d_out

        self.output_layer = nn.Linear(d_in, 1)

    def configure_optimizers(self):
        return self.optimiser_class(self.parameters(), **self.optimiser_kwargs)

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            X = self.activation_fn(layer(X))
        return self.output_layer(X).squeeze(dim=1)

    def _identity_activation_fn(self, X: Tensor) -> Tensor:
        return X

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        print(batch_idx)
        return self._step(batch, batch_idx)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, batch_idx)

    def _step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        X, y = batch
        y_hat = self(X)
        return self.loss_fn(y_hat, y)

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        return self(batch[0])

    def validation_epoch_end(self, step_losses: List[Tensor]) -> Dict[str, Tensor]:
        loss = self._aggregate_losses(step_losses)
        metrics = dict(epoch_val_loss=loss)
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, step_losses: List[Tensor]) -> Dict[str, Tensor]:
        loss = self._aggregate_losses(step_losses)
        metrics = dict(epoch_test_loss=loss)
        self.log_dict(metrics)
        return metrics

    @staticmethod
    def _aggregate_losses(step_losses: List[Tensor]) -> Tensor:
        return torch.stack(step_losses).mean()
