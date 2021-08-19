import pytest
from pytorch_lightning import Trainer

from swafa.models import FeedForwardNet
from swafa.utils import vectorise_weights, get_callback_epoch_range


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
