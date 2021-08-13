from typing import Optional

from torch import Tensor
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

from swafa.custom_types import POSTERIOR_TYPE
from experiments.utils.metrics import compute_distance_between_matrices, compute_gaussian_wasserstein_distance


class PosteriorEvaluationCallback(Callback):
    """
    This callback measures how close the estimated posterior of a model's weights is to the true posterior.

    The callback assumes that the estimated posterior is updated during training and is called at the end of each epoch.

    Args:
        posterior: The estimated posterior distribution of a model's weights. It is assumed that this is updated
            separately during training.
        true_mean: The mean of the true posterior distribution of the same model's weights. Of shape (weight_dim,).
        true_covar: The covariance matrix of the true posterior distribution of the same model's weights. Of shape
            (weight_dim, weight_dim).
        eval_epoch_frequency: The number of epochs between each evaluation of the posterior.

    Attributes:
        eval_epochs: (List[int]) The epochs on which the posterior was evaluated.
        distances_from_mean: (List[float]) The distance between the mean vector of the true posterior and the mean
            vector of the estimated posterior on each evaluation epoch. Measured by the Frobenius norm.
        distances_from_covar: (List[float]) The distance between the covariance matrix of the true posterior and the
            covariance matrix of the estimated posterior on each evaluation epoch. Measured by the Frobenius norm.
        wasserstein_distances: The 2-Wasserstein distance between the true posterior and the estimated posterior on
            each evaluation epoch.
    """

    def __init__(self, posterior: POSTERIOR_TYPE, true_mean: Tensor, true_covar: Tensor, eval_epoch_frequency: int = 1):
        self.posterior = posterior
        self.true_mean = true_mean
        self.true_covar = true_covar
        self.eval_epoch_frequency = eval_epoch_frequency

        self.eval_epochs = []
        self.distances_from_mean = []
        self.distances_from_covar = []
        self.wasserstein_distances = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused: Optional = None):
        """
        Called when the train epoch ends.

        If the current epoch is divisible by the evaluation frequency, compute and store the following:
            - The distance between the mean vector of the true posterior and the mean vector of the estimated posterior.
            - The distance between the covariance matrix of the true posterior and the covariance matrix of the
                estimated posterior.
            - The 2-Wasserstein distance between the true posterior and the estimated posterior.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained. The total dimension of its weights should match the dimension of the
                true and estimated posterior.
            unused: Only present to match the signature of the original method.
        """
        if (self.eval_epoch_frequency > 0) & (trainer.current_epoch % self.eval_epoch_frequency == 0):
            mean = self.posterior.get_mean()
            covar = self.posterior.get_covariance()

            self.eval_epochs.append(trainer.current_epoch)
            self.distances_from_mean.append(
                compute_distance_between_matrices(self.true_mean, mean)
            )
            self.distances_from_covar.append(
                compute_distance_between_matrices(self.true_covar, covar)
            )
            self.wasserstein_distances.append(
                compute_gaussian_wasserstein_distance(self.true_mean, self.true_covar, mean, covar)
            )
