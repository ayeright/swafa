from typing import Any

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor

from swafa.custom_types import POSTERIOR_TYPE


class WeightPosteriorCallback(Callback):

    def __init__(self, posterior: POSTERIOR_TYPE, update_epoch_start: int = 1):
        error_msg = f"update_epoch_start should be a positive integer or a float between 0 and 1, " \
                    f"not {update_epoch_start}"
        if isinstance(update_epoch_start, int) and update_epoch_start < 1:
            raise ValueError(error_msg)
        if isinstance(update_epoch_start, float) and not (0 <= update_epoch_start <= 1):
            raise ValueError(error_msg)

        self.posterior = posterior
        self._update_epoch_start = update_epoch_start

        self.first_update_epoch = None
        self.last_update_epoch = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        self._init_epoch_range(trainer)

    def _init_epoch_range(self, trainer: Trainer):
        if isinstance(self._update_epoch_start, float):
            self._update_epoch_start = int(trainer.max_epochs * self._update_epoch_start)

        self.first_update_epoch = max(self._update_epoch_start - 1, 0)
        self.last_update_epoch = trainer.max_epochs - 1

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int, dataloader_idx: int):
        if self.first_update_epoch <= trainer.current_epoch <= self.last_update_epoch:
            weights = self._vectorise_weights(pl_module)
            self.posterior.update(weights)

    @staticmethod
    def _vectorise_weights(pl_module: LightningModule) -> Tensor:
        return torch.cat([w.data.reshape(-1) for w in pl_module.parameters()])
