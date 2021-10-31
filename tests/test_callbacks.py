import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer

from swafa.models import FeedForwardNet
from swafa.callbacks import WeightPosteriorCallback, FactorAnalysisVariationalInferenceCallback
from swafa.fa import OnlineGradientFactorAnalysis
from swafa.posterior import ModelPosterior
from swafa.utils import get_weight_dimension


class TestWeightPosteriorUpdate:

    @pytest.mark.parametrize(
        "n_samples, batch_size, n_epochs, update_epoch_start, iterate_averaging_window_size, expected_n_updates",
        [
            (32, 4, 5, 1, 1, int(32 / 4) * 5),
            (32, 4, 5, 3, 1, int(32 / 4) * (5 - 2)),
            (32, 4, 8, 0.5, 1, int(32 / 4) * (8 - 3)),
            (32, 4, 9, 0.5, 1, int(32 / 4) * (9 - 3)),
            (32, 4, 5, 1, 2, (int(32 / 4) * 5 / 2)),
            (32, 4, 5, 3, 2, (int(32 / 4) * (5 - 2) / 2)),
            (32, 4, 8, 0.5, 2, (int(32 / 4) * (8 - 3) / 2)),
            (32, 4, 9, 0.5, 2, (int(32 / 4) * (9 - 3) / 2)),
        ]
    )
    def test_posterior_updates(self, n_samples, batch_size, n_epochs, update_epoch_start, iterate_averaging_window_size,
                               expected_n_updates):
        input_dim = 4
        hidden_dims = [8, 8]
        net = FeedForwardNet(input_dim, hidden_dims)

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        callback = WeightPosteriorCallback(
            posterior=model_posterior.weight_posterior,
            update_epoch_start=update_epoch_start,
            iterate_averaging_window_size=iterate_averaging_window_size,
        )

        trainer = Trainer(max_epochs=n_epochs, callbacks=[callback])

        dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        dataloader = DataLoader(dataset, batch_size=4, drop_last=True)

        trainer.fit(model_posterior.model, train_dataloader=dataloader)

        assert model_posterior.weight_posterior.t == expected_n_updates

    @pytest.mark.parametrize("bad_update_epoch_start", [0, -0.1, 1.1])
    def test_init_raises_error_for_bad_update_epoch_start(self, bad_update_epoch_start):
        net = FeedForwardNet(input_dim=10, hidden_dims=[10])

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        with pytest.raises(ValueError):
            WeightPosteriorCallback(
                posterior=model_posterior.weight_posterior,
                update_epoch_start=bad_update_epoch_start,
            )

    def test_raises_error_if_model_and_posterior_do_not_match(self):
        n_samples = 32
        input_dim = 10
        net = FeedForwardNet(input_dim=input_dim, hidden_dims=[10])

        model_posterior = ModelPosterior(
            model=FeedForwardNet(input_dim=input_dim, hidden_dims=[5]),
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        callback = WeightPosteriorCallback(
            posterior=model_posterior.weight_posterior,
            update_epoch_start=1,
        )

        trainer = Trainer(max_epochs=10, callbacks=[callback])

        dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        dataloader = DataLoader(dataset, batch_size=4, drop_last=True)

        with pytest.raises(RuntimeError):
            trainer.fit(net, train_dataloader=dataloader)

    def test_update_weight_window_average(self):
        input_dim = 10
        net = FeedForwardNet(input_dim=input_dim, bias=False)

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        callback = WeightPosteriorCallback(
            posterior=model_posterior.weight_posterior,
            update_epoch_start=1,
        )

        weights1 = torch.randn(input_dim)
        callback._weight_window_average = torch.clone(weights1)
        callback._window_index = 1
        weights2 = torch.randn(input_dim)
        callback._update_weight_window_average(weights2)

        assert torch.isclose(callback._weight_window_average, (weights1 + weights2) / 2).all()
        assert callback._window_index == 2


class TestFactorAnalysisVariationalInferenceCallback:

    @pytest.mark.parametrize(
        "input_dim, hidden_dims, latent_dim",
        [
            (15, None, 1),
            (24, [8], 2),
            (32, [16, 8], 4),
        ]
    )
    def test_variational_distribution_params_shape(self, input_dim, hidden_dims, latent_dim):
        net = FeedForwardNet(input_dim, hidden_dims)
        weight_dim = get_weight_dimension(net)

        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision=0.1, learning_rate=0.1)
        trainer = Trainer(max_epochs=1, callbacks=[callback])

        callback.on_fit_start(trainer, net)

        assert callback.c.shape == (weight_dim, 1)
        assert callback.F.shape == (weight_dim, latent_dim)
        assert callback.diag_psi.shape == (weight_dim, 1)

    @pytest.mark.parametrize(
        "input_dim, hidden_dims, latent_dim",
        [
            (15, None, 1),
            (24, [8], 2),
            (32, [16, 8], 4),
        ]
    )
    def test_sample_weight_vector(self, input_dim, hidden_dims, latent_dim):
        net = FeedForwardNet(input_dim, hidden_dims)

        callback = FactorAnalysisVariationalInferenceCallback(
            latent_dim, precision=0.1, learning_rate=0.1, random_seed=1,
        )
        trainer = Trainer(max_epochs=1, callbacks=[callback])
        callback.on_fit_start(trainer, net)

        samples = torch.hstack([callback._sample_weight_vector()[:, None] for _ in range(10000)]).numpy()
        actual_mean = samples.mean(axis=1, keepdims=True)
        actual_cov = np.cov(samples)

        expected_mean = callback.c.numpy()
        expected_cov = callback.F.mm(callback.F.t()) + torch.diag(callback.diag_psi.squeeze()).numpy()

        assert np.isclose(actual_mean, expected_mean, atol=0.1).all()
        assert np.isclose(actual_cov, expected_cov, atol=0.1).all()

    def test_compute_gradient_wrt_c(self):
        latent_dim = 2
        precision = 0.1
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision, learning_rate=0.1)
        callback.c = torch.tensor([[1, 2, 3]]).t()
        grad_weights = torch.tensor([[-1, 2, 1]]).t()
        expected_grad = torch.tensor([[1.1, -1.8, -0.7]]).t()

        actual_grad = callback._compute_gradient_wrt_c(grad_weights)

        assert torch.isclose(actual_grad, expected_grad).all()

    def test_compute_gradient_wrt_F(self):
        latent_dim = 2
        precision = 0.1
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision, learning_rate=0.1)
        callback.F = torch.tensor([[1, 2], [3, 4], [5, 6]])
        callback._h = torch.tensor([[1, 3]]).t()
        grad_weights = torch.tensor([[-1, 2, 1]]).t()
        expected_grad = torch.tensor([[1.1, 3.2], [-1.7, -5.6], [-0.5, -2.4]])

        actual_grad = callback._compute_gradient_wrt_F(grad_weights)

        assert torch.isclose(actual_grad, expected_grad).all()

    def test_compute_gradient_wrt_log_diag_psi(self):
        latent_dim = 2
        precision = 0.1
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision, learning_rate=0.1)
        callback.diag_psi = torch.tensor([[1, 2, 3]]).t()
        callback._z = torch.tensor([[1, 3, -1]]).t()
        callback._sqrt_diag_psi_dot_z = torch.sqrt(callback.diag_psi) * callback._z
        grad_weights = torch.tensor([[-1, 2, 1]]).t()
        expected_grad = torch.tensor([[0.05, -0.4 - 3 * np.sqrt(2), -0.35 + np.sqrt(3) / 2]]).float().t()

        actual_grad = callback._compute_gradient_wrt_log_diag_psi(grad_weights)

        assert torch.isclose(actual_grad, expected_grad).all()

    def test_perform_sgd_step(self):
        latent_dim = 2
        precision = 0.1
        learning_rate = 0.1
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision, learning_rate)
        param = torch.tensor([[1, 2], [3, 4]])
        grad = torch.tensor([[-1, 1], [2, 1]])
        expected_updated_param = torch.tensor([[1.1, 1.9], [2.8, 3.9]])

        actual_updated_param = callback._perform_sgd_step(param, grad)

        assert torch.isclose(actual_updated_param, expected_updated_param).all()

    @pytest.mark.parametrize(
        "input_dim, hidden_dims, latent_dim, precision, learning_rate",
        [
            (15, None, 1, 0.1, 0.1),
            (24, [8], 2, 0.1, 0.01),
            (32, [16, 8], 4, 1, 0.001),
        ]
    )
    def test_variational_distribution_params_shape(self, input_dim, hidden_dims, latent_dim, precision, learning_rate):
        net = FeedForwardNet(input_dim, hidden_dims)
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision, learning_rate, max_grad_norm=1.0)

        n_samples = 8
        dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.randn(n_samples))
        dataloader = DataLoader(dataset, batch_size=4)

        trainer = Trainer(max_epochs=1, callbacks=[callback])
        trainer.fit(net, train_dataloader=dataloader)

        c_before = torch.clone(callback.c)
        F_before = torch.clone(callback.F)
        diag_psi_before = torch.clone(callback.diag_psi)

        trainer = Trainer(max_epochs=1, callbacks=[callback])
        trainer.fit(net, train_dataloader=dataloader)

        c_after = torch.clone(callback.c)
        F_after = torch.clone(callback.F)
        diag_psi_after = torch.clone(callback.diag_psi)

        assert torch.isclose(callback.diag_psi, torch.exp(callback._log_diag_psi)).all()
        assert not torch.isclose(c_before, c_after).all()
        assert not torch.isclose(F_before, F_after).all()
        assert not torch.isclose(diag_psi_before, diag_psi_after).all()
