from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import Tensor
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.decomposition import FactorAnalysis

from swafa.custom_types import POSTERIOR_TYPE
from swafa.utils import vectorise_weights, get_callback_epoch_range
from experiments.utils.metrics import compute_distance_between_matrices, compute_gaussian_wasserstein_distance


class BasePosteriorEvaluationCallback(Callback, ABC):
    """
    This is an abstract callback which, when fully implemented, can be used to measure how close an estimated posterior
    of a model's weights is to the true posterior.

    Requires implementation of a method which returns the mean and covariance of the estimated posterior.

    Note: This callback does not update the estimated posterior. This must be done separately.

    Args:
        posterior: The estimated posterior distribution of a model's weights.
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

    def __init__(self, posterior: Any, true_mean: Tensor, true_covar: Tensor, eval_epoch_frequency: int = 1):
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
            mean, covar = self.get_mean_and_covariance()

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

    @abstractmethod
    def get_mean_and_covariance(self) -> (Tensor, Tensor):
        """
        Get the mean and covariance of the estimated posterior of the model's weights.

        Returns:
            mean: The mean of the posterior. Of shape (weight_dim,)
            covar: The covariance of the posterior. Of shape (weight_dim, weight_dim).
        """
        ...


class OnlinePosteriorEvaluationCallback(BasePosteriorEvaluationCallback):
    """
    This callback measures how close the estimated posterior of a model's weights is to the true posterior.

    The callback assumes that the estimated posterior is updated online during training and is called at the end of each
    epoch.

    Args:
        posterior: The estimated posterior distribution of a model's weights.
        true_mean: The mean of the true posterior distribution of the same model's weights. Of shape (weight_dim,).
        true_covar: The covariance matrix of the true posterior distribution of the same model's weights. Of shape
            (weight_dim, weight_dim).
        eval_epoch_frequency: The number of epochs between each evaluation of the posterior.
    """

    def __init__(self, posterior: POSTERIOR_TYPE, true_mean: Tensor, true_covar: Tensor, eval_epoch_frequency: int = 1):
        super().__init__(posterior, true_mean, true_covar, eval_epoch_frequency=eval_epoch_frequency)

    def get_mean_and_covariance(self) -> (Tensor, Tensor):
        """
        Get the mean and covariance of the estimated posterior of the model's weights.

        Returns:
            mean: The mean of the posterior. Of shape (weight_dim,)
            covar: The covariance of the posterior. Of shape (weight_dim, weight_dim).
        """
        return self.posterior.get_mean(), self.posterior.get_covariance()


class BatchFactorAnalysisPosteriorEvaluationCallback(BasePosteriorEvaluationCallback):
    """
    This callback measures how close the estimated posterior of a model's weights is to the true posterior.

    The posterior is learned via a batch factor analysis (FA) algorithm (randomised SVD). The model's weights are
    collected after each batch training step. Before each evaluation of the posterior, the batch FA algorithm is fit to
    the weight vectors which have been collected up to that point.

    Args:
        latent_dim: The latent dimension of the batch FA model.
        true_mean: The mean of the true posterior distribution of the same model's weights. Of shape (weight_dim,).
        true_covar: The covariance matrix of the true posterior distribution of the same model's weights. Of shape
            (weight_dim, weight_dim).
        collect_epoch_start: The training epoch on which to start collecting weight vectors. Integer indexing starts
            from 1. Can also specify a float between 0 and 1, which corresponds to the fraction of total epochs which
            should pass before starting to collect weight vectors.
        eval_epoch_frequency: The number of epochs between each evaluation of the posterior.
        random_seed: The random seed used when fitting the FA model.

    Attributes:
        first_collect_epoch: The first epoch on which weight vectors will be collected.
        last_collect_epoch: The last epoch on which weight vectors will be collected.
        weight_iterates: A list of arrays, which are the weight vectors collected during training.
    """

    def __init__(self, latent_dim: int, true_mean: Tensor, true_covar: Tensor,
                 collect_epoch_start: Union[int, float] = 1, eval_epoch_frequency: int = 1, random_seed: int = 0):
        error_msg = f"collect_epoch_start should be a positive integer or a float between 0 and 1, " \
                    f"not {collect_epoch_start}"
        if isinstance(collect_epoch_start, int) and collect_epoch_start < 1:
            raise ValueError(error_msg)
        if isinstance(collect_epoch_start, float) and not (0 <= collect_epoch_start <= 1):
            raise ValueError(error_msg)

        posterior = FactorAnalysis(n_components=latent_dim, svd_method='randomized', random_state=random_seed)
        super().__init__(posterior, true_mean, true_covar, eval_epoch_frequency=eval_epoch_frequency)

        self._collect_epoch_start = collect_epoch_start

        self.first_collect_epoch = None
        self.last_collect_epoch = None
        self.weight_iterates = []

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when fit begins.

        Initialise the range of epochs on which weight vectors will be collected

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        self.first_collect_epoch, self.last_collect_epoch = get_callback_epoch_range(
            trainer, epoch_start=self._collect_epoch_start,
        )

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int, dataloader_idx: int):
        """
        Called when the train batch ends.

        Collect the model's current weight vector.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
            outputs: Not used.
            batch: Not used.
            batch_idx: Not used.
            dataloader_idx: Not used.
        """
        if self.first_collect_epoch <= trainer.current_epoch <= self.last_collect_epoch:
            self.weight_iterates.append(vectorise_weights(pl_module).detach().cpu().numpy())

    def get_mean_and_covariance(self) -> (Tensor, Tensor):
        """
        Fit a batch FA algorithm to the weight vectors and return the mean and covariance of the FA model.

        If no weight vectors have been collected yet, return zero tensors.

        Returns:
            mean: The mean of the posterior. Of shape (weight_dim,)
            covar: The covariance of the posterior. Of shape (weight_dim, weight_dim).
        """
        if len(self.weight_iterates) == 0:
            return torch.zeros_like(self.true_mean), torch.zeros_like(self.true_covar)

        W = np.vstack(self.weight_iterates)
        self.posterior.fit(W)
        mean = torch.from_numpy(self.posterior.mean_).float()
        covar = torch.from_numpy(self.posterior.get_covariance()).float()

        return mean, covar
