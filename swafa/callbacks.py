from typing import Any, Union

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor

from swafa.custom_types import POSTERIOR_TYPE
from swafa.utils import get_callback_epoch_range, vectorise_weights


class WeightPosteriorCallback(Callback):
    """
    A callback which can be used with a PyTorch Lightning Trainer to update the posterior distribution of a model's
    weights.

    The updates are performed using the weight iterates sampled after each mini-batch update. Each iterate can update
    the posterior separately, or alternatively, the update can be made using the average of the iterates within a fixed
    window.

    When this callback is used while training a model, the dimension of the posterior distribution must match the
    dimension of the model's weight space.

    Args:
        posterior: Posterior distribution over the weights of a PyTorch Lighting model.
        update_epoch_start: The training epoch on which to start updating the posterior. Integer indexing starts from 1.
            Can also specify a float between 0 and 1, which corresponds to the fraction of total epochs which should
            pass before starting to update the posterior.
        iterate_averaging_window_size: The size of the window for averaging weight iterates. An update will be made to
            the posterior using each window average. Setting this to 1 is equivalent to using each iterate to update
            the posterior separately.

    Attributes:
        first_update_epoch: The epoch on which the updates to the posterior will start.
        last_update_epoch: The epoch on which the updates to the posterior will end.
    """

    def __init__(self, posterior: POSTERIOR_TYPE, update_epoch_start: Union[int, float] = 1,
                 iterate_averaging_window_size: int = 1):
        error_msg = f"update_epoch_start should be a positive integer or a float between 0 and 1, " \
                    f"not {update_epoch_start}"
        if isinstance(update_epoch_start, int) and update_epoch_start < 1:
            raise ValueError(error_msg)
        if isinstance(update_epoch_start, float) and not (0 <= update_epoch_start <= 1):
            raise ValueError(error_msg)

        self.posterior = posterior
        self._update_epoch_start = update_epoch_start

        self.iterate_averaging_window_size = iterate_averaging_window_size
        self._weight_window_average = None
        self._window_index = 0

        self.first_update_epoch = None
        self.last_update_epoch = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when fit begins.

        Initialise the range of epochs on which the posterior will be updated and check that the dimension of the
        posterior distribution matches the dimension of the model's weight space.

        Also, initialise the average weight vector within the current window.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        self.first_update_epoch, self.last_update_epoch = get_callback_epoch_range(
            trainer, epoch_start=self._update_epoch_start,
        )
        weights = self._check_weight_dimension(pl_module)
        self._reset_weight_window_average(weights)

    def _check_weight_dimension(self, pl_module: LightningModule) -> Tensor:
        """
        Check that the dimension of the posterior distribution matches the dimension of the model's weight space.

        If not, raise a RuntimeError.

        Args:
            pl_module: The model being trained.

        Returns:
            The vectorised model weights, of shape (n_weights,).
        """
        weights = vectorise_weights(pl_module)
        weight_dim = len(weights)
        if weight_dim != self.posterior.observation_dim:
            raise RuntimeError(f"The dimension of the model and the posterior weight distribution must match, but they "
                               f"are {weight_dim} and {self.posterior.observation_dim}, respectively")

        return weights

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int, dataloader_idx: int):
        """
        Called when the train batch ends.

        If within the update epoch range, update the weight iterates window average using the latest setting of the
        model's weights.

        If the weight iterate averaging window size has been reached, use the window average to update the posterior
        distribution.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
            outputs: Not used.
            batch: Not used.
            batch_idx: Not used.
            dataloader_idx: Not used.
        """
        if self.first_update_epoch <= trainer.current_epoch <= self.last_update_epoch:
            weights = vectorise_weights(pl_module)
            self._update_weight_window_average(weights)

            if self._window_index == self.iterate_averaging_window_size:
                self.posterior.update(self._weight_window_average)
                self._reset_weight_window_average(weights)

    def _update_weight_window_average(self, weights: Tensor):
        """
        Increment window index by 1 and update the running average of the window weight iterates.

        Args:
            weights: The vectorised model weights, of shape (n_weights,).
        """
        self._window_index += 1
        self._weight_window_average = \
            self._weight_window_average + (weights - self._weight_window_average) / self._window_index

    def _reset_weight_window_average(self, weights: Tensor):
        """
        Reset the window average of the weight iterates to a tensor of 0s and reset the window index to 0.

        Args:
            weights: The vectorised model weights, of shape (n_weights,).
        """
        self._window_index = 0
        self._weight_window_average = torch.zeros_like(weights)
