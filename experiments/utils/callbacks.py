from typing import Optional

from torch import Tensor
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

from swafa.custom_types import POSTERIOR_TYPE


class PosteriorEvaluationCallback(Callback):

    def __init__(self, posterior: POSTERIOR_TYPE, true_mean: Tensor, true_covar: Tensor, eval_epoch_frequency: int):
        self.posterior = posterior
        self.true_mean = true_mean
        self.true_covar = true_covar
        self.eval_epoch_frequency = eval_epoch_frequency

        self.eval_epochs = []
        self.relative_covar_distance = []
        self.wasserstein_distance = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused: Optional = None):
        if (self.eval_epoch_frequency > 0) & (trainer.current_epoch % self.eval_epoch_frequency == 0):


