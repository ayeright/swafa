import torch
import pytest
from torch.optim import Optimizer, SGD

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

        fa._calc_gradient_wrt_F(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        assert torch.isclose(fa.F.grad, expected_gradient, atol=1e-05).all()

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

        fa._calc_gradient_wrt_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        assert torch.isclose(fa.diag_psi.grad, expected_gradient, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_calc_gradient_wrt_log_psi(self, observation_dim, latent_dim, init_factors_noise_std):
        fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt = self._get_commons(
            observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std
        )

        fa._calc_gradient_wrt_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        expected_gradient = fa.diag_psi.grad * fa.diag_psi

        fa._calc_gradient_wrt_log_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        assert torch.isclose(fa.log_diag_psi.grad, expected_gradient, atol=1e-05).all()

    # @pytest.mark.parametrize("observation_dim", [2])
    # @pytest.mark.parametrize("latent_dim", [3])
    # @pytest.mark.parametrize("init_factors_noise_std", [1e-3])
    # @pytest.mark.parametrize("learning_rate", [1e-3])
    # def test_sgd_update_F(self, observation_dim, latent_dim, init_factors_noise_std, learning_rate):
    #     fa = OnlineGradientFactorAnalysis(
    #         observation_dim, latent_dim, optimiser=SGD, optimiser_kwargs=dict(lr=learning_rate),
    #         init_factors_noise_std=init_factors_noise_std
    #     )
    #     old_F = fa.F.clone()
    #     theta = torch.randn(observation_dim, 1)
    #     fa.update(theta)
    #     expected_new_F = old_F + learning_rate * fa.F.grad
    #
    #     print(expected_new_F)
    #     print(fa.F)
    #
    #     assert torch.isclose(fa.F, expected_new_F, atol=1e-05).all()


    @staticmethod
    def _get_commons(observation_dim: int, latent_dim: int, optimiser: Optimizer = SGD, optimiser_kwargs: dict = None,
                     init_factors_noise_std: float = 1e-3):
        optimiser_kwargs = optimiser_kwargs or dict()
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=optimiser, optimiser_kwargs=optimiser_kwargs,
            init_factors_noise_std=init_factors_noise_std
        )
        fa.c = torch.randn(observation_dim, 1)
        fa.F = torch.randn(observation_dim, latent_dim)
        fa.diag_psi = torch.randn(observation_dim, 1)
        theta = torch.randn(observation_dim, 1)
        d, diag_inv_psi, m, sigma = fa._update_commons(theta)
        inv_psi = torch.diag(diag_inv_psi.squeeze())
        F_times_sigma_plus_m_mt = fa._calc_F_times_sigma_plus_m_mt(m, sigma)
        return fa, d, diag_inv_psi, m, sigma, inv_psi, F_times_sigma_plus_m_mt
