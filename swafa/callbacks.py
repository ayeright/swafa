from typing import Any, Union

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from swafa.custom_types import POSTERIOR_TYPE
from swafa.utils import get_callback_epoch_range, vectorise_weights


class WeightPosteriorCallback(Callback):
    """
    A callback which can be used with a PyTorch Lightning Trainer to update the posterior distribution of a model's
    weights.

    When this callback is used while training a model, the dimension of the posterior distribution must match the
    dimension of the model's weight space.

    Args:
        posterior: Posterior distribution over the weights of a PyTorch Lighting model.
        update_epoch_start: The training epoch on which to start updating the posterior. Integer indexing starts from 1.
            Can also specify a float between 0 and 1, which corresponds to the fraction of total epochs which should
            pass before starting to update the posterior.

    Attributes:
        first_update_epoch: The epoch on which the updates to the posterior will start.
        last_update_epoch: The epoch on which the updates to the posterior will end.
    """

    def __init__(self, posterior: POSTERIOR_TYPE, update_epoch_start: Union[int, float] = 1):
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
        """
        Called when fit begins.

        Initialise the range of epochs on which the posterior will be updated and check that the dimension of the
        posterior distribution matches the dimension of the model's weight space.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        self.first_update_epoch, self.last_update_epoch = get_callback_epoch_range(
            trainer, epoch_start=self._update_epoch_start,
        )
        self._check_weight_dimension(pl_module)

    def _check_weight_dimension(self, pl_module: LightningModule):
        """
        Check that the dimension of the posterior distribution matches the dimension of the model's weight space.

        If not, raise a RuntimeError.

        Args:
            pl_module: The model being trained.
        """
        weight_dim = len(vectorise_weights(pl_module))
        if weight_dim != self.posterior.observation_dim:
            raise RuntimeError(f"The dimension of the model and the posterior weight distribution must match, but they "
                               f"are {weight_dim} and {self.posterior.observation_dim}, respectively")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int, dataloader_idx: int):
        """
        Called when the train batch ends.

        If within the update epoch range, use the latest setting of the model's weights to update the posterior
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
            self.posterior.update(weights)
