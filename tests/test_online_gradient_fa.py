import torch
import pytest
from torch.optim import Optimizer, SGD, Adam

from swafa.fa import OnlineGradientFactorAnalysis


class TestOnlineGradientFactorAnalysis:

    def test_calc_F_times_sigma_plus_m_mt(self):
        fa = OnlineGradientFactorAnalysis(1, 1)
        fa.F = torch.Tensor([[1, 2], [3, 4], [5, 6]])
        sigma = torch.Tensor([[1, 2], [3, 4]])
        m = torch.Tensor([[1], [2]])
        expected_output = torch.Tensor([[12, 20], [26, 44], [40, 68]])
        actual_output = fa._calc_F_times_sigma_plus_m_mt(m, sigma)
        assert torch.isclose(actual_output, expected_output, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_calc_gradient_wrt_F(self, observation_dim, latent_dim, init_factors_noise_std):
        fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt = self._get_commons(
            observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std
        )

        expected_gradient = inv_psi.mm(d.mm(m.t()) - fa.F.mm(sigma + m.mm(m.t())))

        actual_gradient = fa._calc_gradient_wrt_F(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        assert torch.isclose(actual_gradient, expected_gradient, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_calc_gradient_wrt_psi(self, observation_dim, latent_dim, init_factors_noise_std):
        fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt = self._get_commons(
            observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std
        )

        expected_gradient = torch.diag(
            0.5 * (
                torch.diag(
                    torch.diag(inv_psi ** 2) * (
                        d ** 2
                        - 2 * d * (fa.F.mm(m))
                        + torch.diag(fa.F.mm(sigma + m.mm(m.t())).mm(fa.F.t()))
                    )
                )
                - inv_psi
            )
        ).reshape(-1, 1)

        actual_gradient = fa._calc_gradient_wrt_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        assert torch.isclose(actual_gradient, expected_gradient, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_calc_gradient_wrt_log_psi(self, observation_dim, latent_dim, init_factors_noise_std):
        fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt = self._get_commons(
            observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std
        )

        gradient_wrt_diag_psi = fa._calc_gradient_wrt_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        expected_gradient = gradient_wrt_diag_psi * fa.diag_psi

        actual_gradient = fa._calc_gradient_wrt_log_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        assert torch.isclose(actual_gradient, expected_gradient, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_sgd_update_F(self, observation_dim, latent_dim, init_factors_noise_std, learning_rate):
        fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt = self._get_commons(
            observation_dim, latent_dim, optimiser=SGD, optimiser_kwargs=dict(lr=learning_rate),
            init_factors_noise_std=init_factors_noise_std
        )

        old_F = fa.F.clone()
        gradient_wrt_F = fa._calc_gradient_wrt_F(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        gradient_wrt_diag_log_psi = fa._calc_gradient_wrt_log_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        expected_new_F = old_F + learning_rate * gradient_wrt_F

        fa._gradient_step(gradient_wrt_F, gradient_wrt_diag_log_psi)
        assert torch.isclose(fa.F, expected_new_F, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_sgd_update_log_diag_psi(self, observation_dim, latent_dim, init_factors_noise_std, learning_rate):
        fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt = self._get_commons(
            observation_dim, latent_dim, optimiser=SGD, optimiser_kwargs=dict(lr=learning_rate),
            init_factors_noise_std=init_factors_noise_std
        )

        old_log_diag_psi = fa.log_diag_psi.clone()
        gradient_wrt_F = fa._calc_gradient_wrt_F(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        gradient_wrt_diag_log_psi = fa._calc_gradient_wrt_log_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        expected_new_log_diag_psi = old_log_diag_psi + learning_rate * gradient_wrt_diag_log_psi

        fa._gradient_step(gradient_wrt_F, gradient_wrt_diag_log_psi)
        assert torch.isclose(fa.log_diag_psi, expected_new_log_diag_psi, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    @pytest.mark.parametrize("optimiser", [SGD, Adam])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_updates_change_F(self, observation_dim, latent_dim, init_factors_noise_std, optimiser, learning_rate):
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=optimiser, optimiser_kwargs=dict(lr=learning_rate),
            init_factors_noise_std=init_factors_noise_std
        )
        for _ in range(10):
            old_F = fa.F.clone()
            fa.update(torch.randn(observation_dim))
            assert not torch.isclose(fa.F, old_F, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    @pytest.mark.parametrize("optimiser", [SGD, Adam])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_updates_change_log_diag_psi(self, observation_dim, latent_dim, init_factors_noise_std, optimiser,
                                         learning_rate):
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=optimiser, optimiser_kwargs=dict(lr=learning_rate),
            init_factors_noise_std=init_factors_noise_std
        )
        for _ in range(10):
            old_log_diag_psi = fa.log_diag_psi.clone()
            fa.update(torch.randn(observation_dim))
            assert not torch.isclose(fa.log_diag_psi, old_log_diag_psi, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    @pytest.mark.parametrize("optimiser", [SGD, Adam])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_updated_diag_psi(self, observation_dim, latent_dim, init_factors_noise_std, optimiser, learning_rate):
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=optimiser, optimiser_kwargs=dict(lr=learning_rate),
            init_factors_noise_std=init_factors_noise_std
        )
        for _ in range(10):
            fa.update(torch.randn(observation_dim, 1))
            expected_diag_psi = torch.exp(fa.log_diag_psi)
            assert torch.isclose(fa.diag_psi, expected_diag_psi, atol=1e-05).all()

    @staticmethod
    def _get_commons(observation_dim: int, latent_dim: int, optimiser: Optimizer = SGD, optimiser_kwargs: dict = None,
                     init_factors_noise_std: float = 1e-3):
        optimiser_kwargs = optimiser_kwargs or dict(lr=1e-3)
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=optimiser, optimiser_kwargs=optimiser_kwargs,
            init_factors_noise_std=init_factors_noise_std
        )
        fa.c = torch.randn(observation_dim, 1)
        fa.diag_psi = torch.randn(observation_dim, 1)
        theta = torch.randn(observation_dim, 1)
        d, diag_inv_psi, m, sigma = fa._update_commons(theta)
        inv_psi = torch.diag(diag_inv_psi.squeeze())
        F_times_sigma_plus_m_mt = fa._calc_F_times_sigma_plus_m_mt(m, sigma)
        return fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt
