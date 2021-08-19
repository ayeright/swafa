from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
from pytorch_lightning import Trainer


def get_callback_epoch_range(trainer: Trainer, epoch_start: Optional[Union[int, float]] = None,
                             epoch_stop: Optional[Union[int, float]] = None) -> (int, int):
    """
    Initialise the range of epochs on which a callback will be triggered.

    Convert the epoch limits from float to int if necessary and converts to zero-indexing.

    Args:
        trainer: The PyTorch Lightning Trainer which will trigger the callback.
        epoch_start: The first training epoch on which to trigger the callback. Integer indexing starts from 1. Can
            also specify a float between 0 and 1, which corresponds to the fraction of total epochs which should pass
            before triggering the callback for the first time.
        epoch_stop: The last training epoch on which to trigger the callback. Integer indexing starts from 1. Can
            also specify a float between 0 and 1, which corresponds to the fraction of total epochs which should pass
            before triggering the callback for the last time.
    """
    epoch_start = epoch_start or 1
    epoch_stop = epoch_stop or trainer.max_epochs

    if isinstance(epoch_start, float):
        epoch_start = int(trainer.max_epochs * epoch_start)

    if isinstance(epoch_stop, float):
        epoch_stop = int(trainer.max_epochs * epoch_stop)

    first_epoch = max(epoch_start - 1, 0)
    last_epoch = min(epoch_stop - 1, trainer.max_epochs - 1)

    return first_epoch, last_epoch


def vectorise_weights(model: nn.Module) -> Tensor:
    """
    Concatenate all weights of the given model into a single vector.

    Each individual set of weights is reshaped into a single vector and then these vectors are stacked together.

    The weights are stacked in the order that they are returned by model.parameters().

    Args:
        model: A PyTorch model.

    Returns:
        All the model's weights stacked together. Of shape (n_weights,).
    """
    return torch.cat([w.data.reshape(-1) for w in model.parameters()])
