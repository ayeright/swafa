import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from pytorch_lightning import Trainer

from swafa.models import FeedForwardNet
from swafa.callbacks import WeightPosteriorCallback, FactorAnalysisVariationalInferenceCallback
from swafa.fa import OnlineGradientFactorAnalysis
from swafa.posterior import ModelPosterior
from swafa.utils import get_weight_dimension


# class TestWeightPosteriorUpdate:
#
#     @pytest.mark.parametrize(
#         "n_samples, batch_size, n_epochs, update_epoch_start, iterate_averaging_window_size, expected_n_updates",
#         [
#             (32, 4, 5, 1, 1, int(32 / 4) * 5),
#             (32, 4, 5, 3, 1, int(32 / 4) * (5 - 2)),
#             (32, 4, 8, 0.5, 1, int(32 / 4) * (8 - 3)),
#             (32, 4, 9, 0.5, 1, int(32 / 4) * (9 - 3)),
#             (32, 4, 5, 1, 2, (int(32 / 4) * 5 / 2)),
#             (32, 4, 5, 3, 2, (int(32 / 4) * (5 - 2) / 2)),
#             (32, 4, 8, 0.5, 2, (int(32 / 4) * (8 - 3) / 2)),
#             (32, 4, 9, 0.5, 2, (int(32 / 4) * (9 - 3) / 2)),
#         ]
#     )
#     def test_posterior_updates(self, n_samples, batch_size, n_epochs, update_epoch_start, iterate_averaging_window_size,
#                                expected_n_updates):
#         input_dim = 4
#         hidden_dims = [8, 8]
#         net = FeedForwardNet(input_dim, hidden_dims)
#
#         model_posterior = ModelPosterior(
#             model=net,
#             weight_posterior_class=OnlineGradientFactorAnalysis,
#             weight_posterior_kwargs=dict(latent_dim=3),
#         )
#
#         callback = WeightPosteriorCallback(
#             posterior=model_posterior.weight_posterior,
#             update_epoch_start=update_epoch_start,
#             iterate_averaging_window_size=iterate_averaging_window_size,
#         )
#
#         trainer = Trainer(max_epochs=n_epochs, callbacks=[callback])
#
#         dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
#         dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
#
#         trainer.fit(model_posterior.model, train_dataloader=dataloader)
#
#         assert model_posterior.weight_posterior.t == expected_n_updates
#
#     @pytest.mark.parametrize("bad_update_epoch_start", [0, -0.1, 1.1])
#     def test_init_raises_error_for_bad_update_epoch_start(self, bad_update_epoch_start):
#         net = FeedForwardNet(input_dim=10, hidden_dims=[10])
#
#         model_posterior = ModelPosterior(
#             model=net,
#             weight_posterior_class=OnlineGradientFactorAnalysis,
#             weight_posterior_kwargs=dict(latent_dim=3),
#         )
#
#         with pytest.raises(ValueError):
#             WeightPosteriorCallback(
#                 posterior=model_posterior.weight_posterior,
#                 update_epoch_start=bad_update_epoch_start,
#             )
#
#     def test_raises_error_if_model_and_posterior_do_not_match(self):
#         n_samples = 32
#         input_dim = 10
#         net = FeedForwardNet(input_dim=input_dim, hidden_dims=[10])
#
#         model_posterior = ModelPosterior(
#             model=FeedForwardNet(input_dim=input_dim, hidden_dims=[5]),
#             weight_posterior_class=OnlineGradientFactorAnalysis,
#             weight_posterior_kwargs=dict(latent_dim=3),
#         )
#
#         callback = WeightPosteriorCallback(
#             posterior=model_posterior.weight_posterior,
#             update_epoch_start=1,
#         )
#
#         trainer = Trainer(max_epochs=10, callbacks=[callback])
#
#         dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
#         dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
#
#         with pytest.raises(RuntimeError):
#             trainer.fit(net, train_dataloader=dataloader)
#
#     def test_update_weight_window_average(self):
#         input_dim = 10
#         net = FeedForwardNet(input_dim=input_dim, bias=False)
#
#         model_posterior = ModelPosterior(
#             model=net,
#             weight_posterior_class=OnlineGradientFactorAnalysis,
#             weight_posterior_kwargs=dict(latent_dim=3),
#         )
#
#         callback = WeightPosteriorCallback(
#             posterior=model_posterior.weight_posterior,
#             update_epoch_start=1,
#         )
#
#         weights1 = torch.randn(input_dim)
#         callback._weight_window_average = torch.clone(weights1)
#         callback._window_index = 1
#         weights2 = torch.randn(input_dim)
#         callback._update_weight_window_average(weights2)
#
#         assert torch.isclose(callback._weight_window_average, (weights1 + weights2) / 2).all()
#         assert callback._window_index == 2


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

        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision=0.1)
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
            latent_dim, precision=0.1, random_seed=1,
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

    @pytest.mark.parametrize(
        "weight_dim, latent_dim, precision",
        [
            (10, 1, 1.0),
            (10, 2, 0.1),
            (20, 5, 0.01),
        ]
    )
    def test_compute_gradient_wrt_c(self, weight_dim, latent_dim, precision):
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision)

        callback.weight_dim = weight_dim
        callback._init_variational_params()
        callback._update_expected_gradients()
        grad_weights = torch.randn(weight_dim, 1)

        actual_grad = callback._compute_gradient_wrt_c(grad_weights)

        expected_var_grad = 0
        expected_prior_grad = -precision * callback.c
        expected_loss_grad = grad_weights

        expected_grad = expected_var_grad - expected_prior_grad + expected_loss_grad

        assert torch.isclose(actual_grad, expected_grad).all()

    @pytest.mark.parametrize(
        "weight_dim, latent_dim, precision",
        [
            (10, 1, 1.0),
            (10, 2, 0.1),
            (20, 5, 0.01),
        ]
    )
    def test_compute_gradient_wrt_F(self, weight_dim, latent_dim, precision):
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision)

        callback.weight_dim = weight_dim
        callback._init_variational_params()
        callback.diag_psi = torch.rand(weight_dim, 1)
        callback._update_expected_gradients()
        callback._sample_weight_vector()
        grad_weights = torch.randn(weight_dim, 1)

        actual_grad = callback._compute_gradient_wrt_F(grad_weights)

        F = callback.F
        Ft = F.t()
        inv_psi = torch.diag(1 / callback.diag_psi.squeeze())
        I = torch.eye(latent_dim)
        sigma = torch.linalg.inv(I + Ft.mm(inv_psi).mm(F))
        h = callback._h

        A = inv_psi.mm(F)
        B = A.t().mm(F)

        expected_var_grad = A.mm(I - sigma.mm(B) - sigma).mm(B).mm(sigma)
        expected_prior_grad = -precision * F
        expected_loss_grad = grad_weights.mm(h.t())

        expected_grad = expected_var_grad - expected_prior_grad + expected_loss_grad

        assert torch.isclose(actual_grad, expected_grad, atol=1e-4).all()

    @pytest.mark.parametrize(
        "weight_dim, latent_dim, precision",
        [
            (10, 1, 1.0),
            (10, 2, 0.1),
            (20, 5, 0.01),
        ]
    )
    def test_compute_gradient_wrt_log_diag_psi(self, weight_dim, latent_dim, precision):
        callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision)

        callback.weight_dim = weight_dim
        callback._init_variational_params()
        callback.diag_psi = torch.rand(weight_dim, 1)
        callback._update_expected_gradients()
        callback._sample_weight_vector()
        grad_weights = torch.randn(weight_dim, 1)

        actual_grad = callback._compute_gradient_wrt_log_diag_psi(grad_weights)

        diag_psi = callback.diag_psi
        F = callback.F
        Ft = F.t()
        psi = torch.diag(callback.diag_psi.squeeze())
        inv_psi = torch.diag(1 / callback.diag_psi.squeeze())
        I = torch.eye(latent_dim)
        sigma = torch.linalg.inv(I + Ft.mm(inv_psi).mm(F))

        A = inv_psi.mm(F)
        B = A.t().mm(F)
        C = sigma.mm(2 * B + I - (B + I).mm(B.mm(sigma))).mm(Ft)
        s = torch.diag(F.mm(Ft) + psi - F.mm(C))[:, None]
        z = callback._z

        expected_var_grad = (1 / 2) * torch.diag(inv_psi)[:, None] * s - (1 / 2)
        expected_prior_grad = (-precision / 2) * diag_psi
        expected_loss_grad = (1 / 2) * torch.diag(grad_weights.mm(z.t()).mm(torch.sqrt(psi)))[:, None]

        expected_grad = expected_var_grad - expected_prior_grad + expected_loss_grad

        assert torch.isclose(actual_grad, expected_grad, atol=1e-4).all()

    @pytest.mark.parametrize(
        "weight_dim, latent_dim, precision, n_gradients_per_update",
        [
            (10, 1, 1.0, 1),
            (10, 2, 0.1, 2),
            (20, 5, 0.01, 3),
        ]
    )
    def test_accumulate_gradients(self, weight_dim, latent_dim, precision, n_gradients_per_update):
        callback = FactorAnalysisVariationalInferenceCallback(
            latent_dim, precision, n_gradients_per_update=n_gradients_per_update,
        )

        callback.weight_dim = weight_dim
        callback._init_variational_params()
        callback._update_expected_gradients()
        callback._sample_weight_vector()
        grad_weights = torch.randn(weight_dim, 1)

        grad_wrt_c = callback._compute_gradient_wrt_c(grad_weights)
        grad_wrt_F = callback._compute_gradient_wrt_F(grad_weights)
        grad_wrt_log_diag_psi = callback._compute_gradient_wrt_log_diag_psi(grad_weights)

        for _ in range(n_gradients_per_update):
            callback._accumulate_gradients(grad_weights)

        assert torch.isclose(callback.c.grad, grad_wrt_c * n_gradients_per_update).all()
        assert torch.isclose(callback.F.grad, grad_wrt_F * n_gradients_per_update).all()
        assert torch.isclose(callback._log_diag_psi.grad, grad_wrt_log_diag_psi * n_gradients_per_update).all()

    @pytest.mark.parametrize(
        "weight_dim, latent_dim, precision, n_gradients_per_update",
        [
            (10, 1, 1.0, 1),
            (10, 2, 0.1, 2),
            (20, 5, 0.01, 3),
        ]
    )
    def test_average_and_normalise_gradient(self, weight_dim, latent_dim, precision, n_gradients_per_update):
        callback = FactorAnalysisVariationalInferenceCallback(
            latent_dim, precision, n_gradients_per_update=n_gradients_per_update,
        )

        callback.weight_dim = weight_dim
        callback._init_variational_params()
        callback._update_expected_gradients()
        callback._sample_weight_vector()
        grad_weights = torch.randn(weight_dim, 1)

        grad_wrt_c = callback._compute_gradient_wrt_c(grad_weights)

        for _ in range(n_gradients_per_update):
            callback._accumulate_gradients(grad_weights)

        callback._average_and_normalise_gradient(callback.c)

        assert torch.isclose(callback.c.grad, grad_wrt_c).all()

    @pytest.mark.parametrize(
        "weight_dim, latent_dim, precision, n_gradients_per_update, learning_rate",
        [
            (10, 1, 1.0, 1, 0.1),
            (10, 2, 0.1, 2, 0.01),
            (20, 5, 0.01, 3, 0.001),
        ]
    )
    def test_update_variational_params(self, weight_dim, latent_dim, precision, n_gradients_per_update, learning_rate):
        callback = FactorAnalysisVariationalInferenceCallback(
            latent_dim, precision, optimiser_class=SGD, optimiser_kwargs=dict(lr=learning_rate),
            n_gradients_per_update=n_gradients_per_update,
        )

        callback.weight_dim = weight_dim
        callback._init_variational_params()
        callback._update_expected_gradients()
        callback._init_optimiser()
        callback._sample_weight_vector()
        grad_weights = torch.randn(weight_dim, 1)

        c_before = torch.clone(callback.c)
        F_before = torch.clone(callback.F)
        log_diag_psi_before = torch.clone(callback._log_diag_psi)

        grad_wrt_c = callback._compute_gradient_wrt_c(grad_weights)
        grad_wrt_F = callback._compute_gradient_wrt_F(grad_weights)
        grad_wrt_log_diag_psi = callback._compute_gradient_wrt_log_diag_psi(grad_weights)

        for _ in range(n_gradients_per_update):
            callback._accumulate_gradients(grad_weights)

        callback._update_variational_params()

        assert torch.isclose(callback.c, c_before - learning_rate * grad_wrt_c).all()
        assert torch.isclose(callback.F, F_before - learning_rate * grad_wrt_F).all()
        assert torch.isclose(callback._log_diag_psi, log_diag_psi_before - learning_rate * grad_wrt_log_diag_psi).all()

        assert torch.isclose(callback.diag_psi, torch.exp(callback._log_diag_psi)).all()

        assert (callback.c.grad == 0).all()
        assert (callback.F.grad == 0).all()
        assert (callback._log_diag_psi.grad == 0).all()

    @pytest.mark.parametrize(
        "input_dim, hidden_dims, latent_dim, precision, learning_rate",
        [
            (15, None, 1, 0.1, 0.1),
            (24, [8], 2, 0.1, 0.01),
            (32, [16, 8], 4, 1, 0.001),
        ]
    )
    def test_variational_distribution_params_change(self, input_dim, hidden_dims, latent_dim, precision, learning_rate):
        net = FeedForwardNet(input_dim, hidden_dims)
        callback = FactorAnalysisVariationalInferenceCallback(
            latent_dim, precision, optimiser_kwargs=dict(lr=learning_rate),
        )

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
