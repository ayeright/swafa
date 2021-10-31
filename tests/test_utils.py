import pytest
from pytorch_lightning import Trainer
import torch

from swafa.models import FeedForwardNet
from swafa.utils import (
    get_callback_epoch_range,
    vectorise_weights,
    vectorise_gradients,
    set_weights,
    get_weight_dimension,
    normalise_gradient,
)


@pytest.mark.parametrize(
    "max_epochs, epoch_start, epoch_stop, expected_first_epoch, expected_last_epoch",
    [
        (10, 1, 10, 0, 9),
        (10, 1, None, 0, 9),
        (10, None, 10, 0, 9),
        (10, None, None, 0, 9),
        (20, 0.0, 1.0, 0, 19),
        (20, 0.2, 0.8, 3, 15),
    ]
)
def test_get_callback_epoch_range(max_epochs, epoch_start, epoch_stop, expected_first_epoch, expected_last_epoch):
    trainer = Trainer(max_epochs=max_epochs)
    first_epoch, last_epoch = get_callback_epoch_range(trainer, epoch_start=epoch_start, epoch_stop=epoch_stop)

    assert first_epoch == expected_first_epoch
    assert last_epoch == expected_last_epoch


@pytest.mark.parametrize(
    "input_dim, hidden_dims, expected_n_weights",
    [
        (5, None, 5 + 1),
        (6, [4], (6 + 1) * 4 + (4 + 1)),
        (7, [6, 9], (7 + 1) * 6 + (6 + 1) * 9 + (9 + 1)),
    ]
)
def test_vectorise_weights(input_dim, hidden_dims, expected_n_weights):
    net = FeedForwardNet(input_dim, hidden_dims)
    weights = vectorise_weights(net)

    assert len(weights) == expected_n_weights


@pytest.mark.parametrize(
    "input_dim, hidden_dims, expected_n_gradients",
    [
        (5, None, 5 + 1),
        (6, [4], (6 + 1) * 4 + (4 + 1)),
        (7, [6, 9], (7 + 1) * 6 + (6 + 1) * 9 + (9 + 1)),
    ]
)
def test_vectorise_gradients(input_dim, hidden_dims, expected_n_gradients):
    net = FeedForwardNet(input_dim, hidden_dims)
    x = torch.randn(3, input_dim)
    loss = net(x).sum()
    loss.backward()
    gradients = vectorise_gradients(net)

    assert len(gradients) == expected_n_gradients


@pytest.mark.parametrize(
    "input_dim, hidden_dims, expected_weight_dim",
    [
        (5, None, 5 + 1),
        (6, [4], (6 + 1) * 4 + (4 + 1)),
        (7, [6, 9], (7 + 1) * 6 + (6 + 1) * 9 + (9 + 1)),
    ]
)
def test_get_weight_dimension(input_dim, hidden_dims, expected_weight_dim):
    net = FeedForwardNet(input_dim, hidden_dims)

    assert get_weight_dimension(net) == expected_weight_dim


@pytest.mark.parametrize(
    "input_dim, hidden_dims",
    [
        (5, None),
        (6, [4]),
        (7, [6, 9]),
    ]
)
def test_set_weights(input_dim, hidden_dims):
    net = FeedForwardNet(input_dim, hidden_dims)
    original_weights = vectorise_weights(net)
    n_weights = get_weight_dimension(net)
    expected_weights = torch.randn(n_weights)
    set_weights(net, expected_weights)
    actual_weights = vectorise_weights(net)

    assert not torch.isclose(actual_weights, original_weights).all()
    assert torch.isclose(actual_weights, expected_weights).all()


@pytest.mark.parametrize(
    "grad, max_grad_norm, expected_grad_norm",
    [
        (torch.tensor([1, 1]).float(), 100, torch.sqrt(torch.tensor(2))),
        (torch.tensor([10, 10]).float(), 5, torch.tensor(5).float()),
    ]
)
def test_set_weights(grad, max_grad_norm, expected_grad_norm):
    actual_grad_norm = torch.linalg.norm(normalise_gradient(grad, max_grad_norm))

    assert torch.isclose(actual_grad_norm, expected_grad_norm)
