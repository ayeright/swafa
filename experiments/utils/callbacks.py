from typing import Optional

from torch import Tensor
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

from swafa.custom_types import POSTERIOR_TYPE
from experiments.utils.metrics import compute_distance_between_matrices, compute_gaussian_wasserstein_distance


class PosteriorEvaluationCallback(Callback):

    def __init__(self, posterior: POSTERIOR_TYPE, true_mean: Tensor, true_covar: Tensor, eval_epoch_frequency: int = 1):
        self.posterior = posterior
        self.true_mean = true_mean
        self.true_covar = true_covar
        self.eval_epoch_frequency = eval_epoch_frequency

        self.eval_epochs = []
        self.relative_mean_distances = []
        self.relative_covar_distances = []
        self.wasserstein_distances = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused: Optional = None):
        if (self.eval_epoch_frequency > 0) & (trainer.current_epoch % self.eval_epoch_frequency == 0):
            mean = self.posterior.get_mean()
            covar = self.posterior.get_covariance()

            self.eval_epochs.append(trainer.current_epoch)
            self.relative_mean_distances.append(
                compute_distance_between_matrices(self.true_mean, mean)
            )
            self.relative_covar_distances.append(
                compute_distance_between_matrices(self.true_covar, covar)
            )
            self.wasserstein_distances.append(
                compute_gaussian_wasserstein_distance(self.true_mean, self.true_covar, mean, covar)
            )
