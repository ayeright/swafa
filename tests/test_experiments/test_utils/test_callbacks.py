from typing import List

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

from swafa.models import FeedForwardNet
from swafa.fa import OnlineGradientFactorAnalysis
from swafa.posterior import ModelPosterior
from swafa.utils import get_weight_dimension
from experiments.utils.callbacks import (
    OnlinePosteriorEvaluationCallback,
    BatchFactorAnalysisPosteriorEvaluationCallback,
)
from experiments.utils.metrics import (
    compute_distance_between_matrices,
    compute_gaussian_wasserstein_distance,
)


class TestBasePosteriorEvaluationCallback:

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
        n_samples = 20
        input_dim = 4
        hidden_dims = [8, 8]

        callback, true_mean, true_covar, model_posterior = _init_model_with_online_posterior_evaluation_callback(
            input_dim, hidden_dims, eval_epoch_frequency,
        )
        _fit_model_with_callback(model_posterior.model, callback, n_samples, input_dim, n_epochs)

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
        n_samples = 20
        input_dim = 4
        hidden_dims = [8, 8]

        callback, true_mean, true_covar, model_posterior = _init_model_with_online_posterior_evaluation_callback(
            input_dim, hidden_dims, eval_epoch_frequency,
        )
        _fit_model_with_callback(model_posterior.model, callback, n_samples, input_dim, n_epochs)

        actual_mean, actual_covar = callback.get_mean_and_covariance()
        assert callback.distances_from_mean == [
            compute_distance_between_matrices(true_mean, actual_mean)
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
        n_samples = 20
        input_dim = 4
        hidden_dims = [8, 8]

        callback, true_mean, true_covar, model_posterior = _init_model_with_online_posterior_evaluation_callback(
            input_dim, hidden_dims, eval_epoch_frequency,
        )
        _fit_model_with_callback(model_posterior.model, callback, n_samples, input_dim, n_epochs)

        actual_mean, actual_covar = callback.get_mean_and_covariance()
        assert callback.distances_from_covar == [
            compute_distance_between_matrices(true_covar, actual_covar)
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
        n_samples = 20
        input_dim = 4
        hidden_dims = [8, 8]

        callback, true_mean, true_covar, model_posterior = _init_model_with_online_posterior_evaluation_callback(
            input_dim, hidden_dims, eval_epoch_frequency,
        )
        _fit_model_with_callback(model_posterior.model, callback, n_samples, input_dim, n_epochs)

        actual_distances = [x if not np.isnan(x) else -1 for x in callback.wasserstein_distances]

        actual_mean, actual_covar = callback.get_mean_and_covariance()
        expected_distances = [
            compute_gaussian_wasserstein_distance(true_mean, true_covar, actual_mean, actual_covar)
            for i in range(n_epochs) if i % eval_epoch_frequency == 0
        ]
        expected_distances = [x if not np.isnan(x) else -1 for x in expected_distances]

        assert actual_distances == expected_distances


class TestOnlinePosteriorEvaluationCallback:

    @pytest.mark.parametrize("input_dim", [2, 4])
    @pytest.mark.parametrize("hidden_dims", [[2], [4, 4]])
    def test_get_mean_and_covariance(self, input_dim, hidden_dims):
        eval_epoch_frequency = 1

        callback, true_mean, true_covar, model_posterior = _init_model_with_online_posterior_evaluation_callback(
            input_dim, hidden_dims, eval_epoch_frequency,
        )

        actual_mean, actual_covar = callback.get_mean_and_covariance()

        assert isinstance(actual_mean, Tensor)
        assert isinstance(actual_covar, Tensor)
        assert actual_mean.shape == true_mean.shape
        assert actual_covar.shape == actual_covar.shape
        assert (torch.diag(actual_covar) > 0).all()


class TestBatchFactorAnalysisPosteriorEvaluationCallback:

    @pytest.mark.parametrize(
        "n_samples, batch_size, n_epochs, collect_epoch_start, eval_epoch_frequency, expected_n_weight_iterates",
        [
            (32, 4, 5, 1, 1, int(32 / 4) * 5),
            (32, 4, 5, 3, 2, int(32 / 4) * (5 - 2)),
            (32, 4, 8, 0.5, 1, int(32 / 4) * (8 - 3)),
            (32, 4, 9, 0.5, 2, int(32 / 4) * (9 - 3)),
        ]
    )
    def test_n_weight_iterates(self, n_samples, batch_size, n_epochs, collect_epoch_start, eval_epoch_frequency,
                               expected_n_weight_iterates):
        input_dim = 4
        hidden_dims = [8, 8]

        callback, true_mean, true_covar, model = _init_model_with_batch_factor_analysis_evaluation_callback(
            input_dim, hidden_dims, collect_epoch_start, eval_epoch_frequency,
        )
        _fit_model_with_callback(model, callback, n_samples, input_dim, n_epochs)

        assert len(callback.weight_iterates) == expected_n_weight_iterates

    @pytest.mark.parametrize(
        "input_dim, hidden_dims, n_epochs, collect_epoch_start, eval_epoch_frequency",
        [
            (2, [2], 2, 1, 1),
            (4, [4, 4], 2, 1, 1),
            (2, [2], 2, 3, 1),
            (4, [4, 4], 2, 3, 1),
        ]
    )
    def test_get_mean_and_covariance(self, input_dim, hidden_dims, n_epochs, collect_epoch_start, eval_epoch_frequency):
        n_samples = 20

        callback, true_mean, true_covar, model = _init_model_with_batch_factor_analysis_evaluation_callback(
            input_dim, hidden_dims, collect_epoch_start, eval_epoch_frequency,
        )
        _fit_model_with_callback(model, callback, n_samples, input_dim, n_epochs)

        actual_mean, actual_covar = callback.get_mean_and_covariance()

        assert isinstance(actual_mean, Tensor)
        assert isinstance(actual_covar, Tensor)
        assert actual_mean.shape == true_mean.shape
        assert actual_covar.shape == actual_covar.shape

        if collect_epoch_start <= eval_epoch_frequency:
            assert (torch.diag(actual_covar) > 0).all()
        else:
            assert torch.isclose(actual_mean, torch.zeros_like(actual_mean)).all()
            assert torch.isclose(actual_covar, torch.zeros_like(actual_covar)).all()


def _init_model_with_online_posterior_evaluation_callback(
        input_dim: int, hidden_dims: List[int], eval_epoch_frequency: int,
) -> (OnlinePosteriorEvaluationCallback, Tensor, Tensor, ModelPosterior):
    net = FeedForwardNet(input_dim, hidden_dims)

    model_posterior = ModelPosterior(
        model=net,
        weight_posterior_class=OnlineGradientFactorAnalysis,
        weight_posterior_kwargs=dict(latent_dim=2),
    )

    weight_dim = model_posterior._get_weight_dimension()
    true_mean = torch.randn(weight_dim)
    true_covar = torch.rand(weight_dim, weight_dim)

    callback = OnlinePosteriorEvaluationCallback(
        posterior=model_posterior.weight_posterior,
        true_mean=true_mean,
        true_covar=true_covar,
        eval_epoch_frequency=eval_epoch_frequency
    )

    return callback, true_mean, true_covar, model_posterior


def _init_model_with_batch_factor_analysis_evaluation_callback(
        input_dim: int, hidden_dims: List[int], collect_epoch_start: int, eval_epoch_frequency: int,
) -> (OnlinePosteriorEvaluationCallback, Tensor, Tensor, LightningModule):
    net = FeedForwardNet(input_dim, hidden_dims)

    weight_dim = get_weight_dimension(net)
    true_mean = torch.randn(weight_dim)
    true_covar = torch.rand(weight_dim, weight_dim)

    callback = BatchFactorAnalysisPosteriorEvaluationCallback(
        latent_dim=2,
        true_mean=true_mean,
        true_covar=true_covar,
        collect_epoch_start=collect_epoch_start,
        eval_epoch_frequency=eval_epoch_frequency
    )

    return callback, true_mean, true_covar, net


def _fit_model_with_callback(model: LightningModule, callback: Callback, n_samples: int, input_dim: int, n_epochs: int):
    trainer = Trainer(max_epochs=n_epochs, callbacks=[callback])

    dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.randn(n_samples))
    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)

    trainer.fit(model, train_dataloader=dataloader)
