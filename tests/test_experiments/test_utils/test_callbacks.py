import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer

from swafa.models import FeedForwardNet
from swafa.fa import OnlineGradientFactorAnalysis
from swafa.posterior import ModelPosterior
from experiments.utils.callbacks import PosteriorEvaluationCallback
from experiments.utils.metrics import (
    compute_distance_between_matrices,
    compute_gaussian_wasserstein_distance,
)


class TestPosteriorEvaluationCallback:

    @pytest.mark.parametrize(
        "n_epochs, eval_epoch_frequency",
        [
            (1, 1),
            (2, 1),
            (10, 2),
            (5, 2),
        ]
    )
    def test_eval_epochs(self, n_epochs, eval_epoch_frequency):
        callback, _, _, _ = _fit_model_with_callback(n_epochs, eval_epoch_frequency)
        assert callback.eval_epochs == [i for i in range(n_epochs) if i % eval_epoch_frequency == 0]

    @pytest.mark.parametrize(
        "n_epochs, eval_epoch_frequency",
        [
            (1, 1),
            (2, 1),
            (10, 2),
            (5, 2),
        ]
    )
    def test_distances_from_mean(self, n_epochs, eval_epoch_frequency):
        callback, true_mean, _, posterior = _fit_model_with_callback(n_epochs, eval_epoch_frequency)
        assert callback.distances_from_mean == [
            compute_distance_between_matrices(true_mean, posterior.get_mean())
            for i in range(n_epochs) if i % eval_epoch_frequency == 0
        ]

    @pytest.mark.parametrize(
        "n_epochs, eval_epoch_frequency",
        [
            (1, 1),
            (2, 1),
            (10, 2),
            (5, 2),
        ]
    )
    def test_distances_from_covar(self, n_epochs, eval_epoch_frequency):
        callback, _, true_covar, posterior = _fit_model_with_callback(n_epochs, eval_epoch_frequency)
        assert callback.distances_from_covar == [
            compute_distance_between_matrices(true_covar, posterior.get_covariance())
            for i in range(n_epochs) if i % eval_epoch_frequency == 0
        ]

    @pytest.mark.parametrize(
        "n_epochs, eval_epoch_frequency",
        [
            (1, 1),
            (2, 1),
            (10, 2),
            (5, 2),
        ]
    )
    def test_wasserstein_distances(self, n_epochs, eval_epoch_frequency):
        callback, true_mean, true_covar, posterior = _fit_model_with_callback(n_epochs, eval_epoch_frequency)
        actual_distances = [x if not np.isnan(x) else -1 for x in callback.wasserstein_distances]

        expected_distances = [
            compute_gaussian_wasserstein_distance(
                true_mean, true_covar, posterior.get_mean(), posterior.get_covariance()
            )
            for i in range(n_epochs) if i % eval_epoch_frequency == 0
        ]
        expected_distances = [x if not np.isnan(x) else -1 for x in expected_distances]

        assert actual_distances == expected_distances


def _fit_model_with_callback(n_epochs: int, eval_epoch_frequency: int) -> (PosteriorEvaluationCallback, Tensor, Tensor):
    n_samples = 50
    input_dim = 4
    hidden_dims = [8, 8]
    net = FeedForwardNet(input_dim, hidden_dims)

    model_posterior = ModelPosterior(
        model=net,
        weight_posterior_class=OnlineGradientFactorAnalysis,
        weight_posterior_kwargs=dict(latent_dim=2),
    )

    weight_dim = model_posterior._get_weight_dimension()
    true_mean = torch.randn(weight_dim)
    true_covar = torch.rand(weight_dim, weight_dim)

    callback = PosteriorEvaluationCallback(
        posterior=model_posterior.weight_posterior,
        true_mean=true_mean,
        true_covar=true_covar,
        eval_epoch_frequency=eval_epoch_frequency
    )

    trainer = Trainer(max_epochs=n_epochs, callbacks=[callback])

    dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.randn(n_samples))
    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)

    trainer.fit(model_posterior.model, train_dataloader=dataloader)

    return callback, true_mean, true_covar, model_posterior.weight_posterior
