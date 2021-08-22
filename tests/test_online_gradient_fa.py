import pytest
import torch
from torch.optim import Optimizer, SGD

from swafa.fa import OnlineGradientFactorAnalysis


class TestOnlineGradientFactorAnalysis:

    def test_update_F_times_sigma_plus_m_mt(self):
        fa = OnlineGradientFactorAnalysis(1, 1)
        fa.F = torch.Tensor([[1, 2], [3, 4], [5, 6]])
        fa._sigma = torch.Tensor([[1, 2], [3, 4]])
        fa._m = torch.Tensor([[1], [2]])
        expected_output = torch.Tensor([[12, 20], [26, 44], [40, 68]])
        fa._update_F_times_sigma_plus_m_mt()
        assert torch.isclose(fa._F_times_sigma_plus_m_mt, expected_output, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_gradient_wrt_F(self, observation_dim, latent_dim):
        fa, theta, inv_psi = self._update_commons(observation_dim, latent_dim)

        expected_gradient = inv_psi.mm(fa._d.mm(fa._m.t()) - fa.F.mm(fa._sigma + fa._m.mm(fa._m.t())))

        fa._update_gradient_wrt_F()
        assert torch.isclose(fa._gradient_wrt_F, expected_gradient, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_gradient_wrt_psi(self, observation_dim, latent_dim):
        fa, theta, inv_psi = self._update_commons(observation_dim, latent_dim)

        expected_gradient = torch.diag(
            0.5 * (
                torch.diag(
                    torch.diag(inv_psi ** 2) * (
                        fa._d ** 2
                        - 2 * fa._d * (fa.F.mm(fa._m))
                        + torch.diag(fa.F.mm(fa._sigma + fa._m.mm(fa._m.t())).mm(fa.F.t()))
                    )
                )
                - inv_psi
            )
        ).reshape(-1, 1)

        fa._update_gradient_wrt_psi()
        assert torch.isclose(fa._gradient_wrt_diag_psi, expected_gradient, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_gradient_wrt_log_psi(self, observation_dim, latent_dim):
        fa, theta, inv_psi = self._update_commons(observation_dim, latent_dim)
        fa._update_gradient_wrt_log_psi()
        expected_gradient = fa._gradient_wrt_diag_psi * fa.diag_psi
        assert torch.isclose(fa._gradient_wrt_log_diag_psi, expected_gradient, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_sgd_update_F(self, observation_dim, latent_dim, learning_rate):
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=SGD, optimiser_kwargs=dict(lr=learning_rate),
        )
        for _ in range(10):
            old_F = fa.F.clone()
            fa.update(torch.randn(observation_dim))
            expected_new_F = old_F + learning_rate * fa._gradient_wrt_F
            assert torch.isclose(fa.F, expected_new_F, atol=1e-05).all()
            assert not torch.isclose(fa.F, old_F, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_sgd_update_log_diag_psi(self, observation_dim, latent_dim, learning_rate):
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=SGD, optimiser_kwargs=dict(lr=learning_rate),
        )
        for _ in range(10):
            old_log_diag_psi = fa._log_diag_psi.clone()
            fa.update(torch.randn(observation_dim))
            expected_new_log_diag_psi = old_log_diag_psi + learning_rate * fa._gradient_wrt_log_diag_psi
            assert torch.isclose(fa._log_diag_psi, expected_new_log_diag_psi, atol=1e-05).all()
            assert not torch.isclose(fa._log_diag_psi, old_log_diag_psi, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-1])
    def test_updated_diag_psi(self, observation_dim, latent_dim, learning_rate):
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser_kwargs=dict(lr=learning_rate),
        )
        for _ in range(10):
            old_diag_psi = fa.diag_psi.clone()
            fa.update(torch.randn(observation_dim, 1))
            expected_diag_psi = torch.exp(fa._log_diag_psi)
            assert torch.isclose(fa.diag_psi, expected_diag_psi, atol=1e-05).all()
            assert not torch.isclose(fa.diag_psi, old_diag_psi, atol=1e-05).all()

    @staticmethod
    def _update_commons(observation_dim: int, latent_dim: int, optimiser: Optimizer = SGD,
                        optimiser_kwargs: dict = None):
        optimiser_kwargs = optimiser_kwargs or dict(lr=1e-3)
        fa = OnlineGradientFactorAnalysis(
            observation_dim, latent_dim, optimiser=optimiser, optimiser_kwargs=optimiser_kwargs,
        )
        fa.c = torch.randn(observation_dim, 1)
        fa.diag_psi = torch.randn(observation_dim, 1)
        theta = torch.randn(observation_dim, 1)
        fa._update_commons(theta)
        fa._update_F_times_sigma_plus_m_mt()
        inv_psi = torch.diag(fa._diag_inv_psi.squeeze())
        return fa, theta, inv_psi
