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


def vectorise_gradients(model: nn.Module) -> Tensor:
    """
    Concatenate all gradients of the given model's weights into a single vector.

    Each individual set of gradients is reshaped into a single vector and then these vectors are stacked together.

    The gradients are stacked in the order that the weights are returned by model.parameters().

    Args:
        model: A PyTorch model.

    Returns:
        All the model's gradients stacked together. Of shape (n_weights,).
    """
    return torch.cat([w.grad.reshape(-1) for w in model.parameters()])


def get_weight_dimension(model: nn.Module) -> int:
    """
    Get the total combined dimension of all the weights in the model.

    Returns:
        The total dimension of the model's weights.
    """
    return sum([w.numel() for w in model.parameters()])


def set_weights(model: nn.Module, weights: torch.Tensor):
    """
    Set the learnable parameters of the given model to the given weights.

    The order of the given weights should be the same as that returned by vectorise_weights().

    Args:
        model: A PyTorch model with n_weights learnable parameters.
        weights: A version of the model's weights stacked together. Of shape (n_weights,).
    """
    weight_count = 0
    for w in model.parameters():
        n_elements = w.numel()
        elements = weights[weight_count:weight_count + n_elements]
        w.data = elements.reshape(w.shape)
        weight_count += n_elements


def normalise_gradient(grad: Tensor, max_grad_norm: float) -> Tensor:
    """
    If the gradient norm is greater than the maximum value, normalise the gradient such that its norm is equal to the
    maximum value.

    Args:
        grad: The gradient.
        max_grad_norm: The maximum allowed gradient norm.

    Returns:
        The normalised gradient.
    """
    grad_norm = torch.linalg.norm(grad)
    if grad_norm > max_grad_norm:
        return max_grad_norm * grad / grad_norm

    return grad
