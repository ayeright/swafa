import pytest
import torch

from swafa.fa import OnlineEMFactorAnalysis


class TestOnlineEMFactorAnalysis:

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_B_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        m_times_mt = []
        for _ in range(10):
            fa.update(torch.randn(observation_dim))
            m_times_mt.append(fa._m.mm(fa._m.t()))

        expected_B_hat = torch.dstack(m_times_mt).mean(dim=2)
        assert torch.isclose(fa._B_hat, expected_B_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_H_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        for _ in range(10):
            fa.update(torch.randn(observation_dim, 1))
            expected_H_hat = fa._sigma + fa._B_hat
            assert torch.isclose(fa._H_hat, expected_H_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_A_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        d_times_mt = []
        for _ in range(10):
            fa.update(torch.randn(observation_dim))
            d_times_mt.append(fa._d.mm(fa._m.t()))

        expected_A_hat = torch.dstack(d_times_mt).mean(dim=2)
        assert torch.isclose(fa._A_hat, expected_A_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_d_squared_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        d_squared = []
        for _ in range(10):
            fa.update(torch.randn(observation_dim))
            d_squared.append(fa._d ** 2)

        expected_d_squared_hat = torch.dstack(d_squared).mean(dim=2)
        assert torch.isclose(fa._d_squared_hat, expected_d_squared_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_F(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        for i in range(10):
            old_F = fa.F.clone()
            fa.update(torch.randn(observation_dim))
            if i == 0:
                assert torch.isclose(fa.F, old_F, atol=1e-05).all()
            else:
                expected_F = fa._A_hat.mm(torch.linalg.inv(fa._H_hat))
                assert torch.isclose(fa.F, expected_F, atol=1e-05).all()
                assert not torch.isclose(fa.F, old_F, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_psi(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        d_times_dt = []
        for i in range(10):
            old_diag_psi = fa.diag_psi.clone()
            fa.update(torch.randn(observation_dim))
            d_times_dt.append(fa._d.mm(fa._d.t()))
            if i == 0:
                assert torch.isclose(fa.diag_psi, old_diag_psi, atol=1e-05).all()
            else:
                expected_diag_psi = torch.diag(
                    torch.dstack(d_times_dt).mean(dim=2)
                    - 2 * fa.F.mm(fa._A_hat.t())
                    + fa.F.mm(fa._H_hat).mm(fa.F.t())
                ).reshape(-1, 1)

                assert torch.isclose(fa.diag_psi, expected_diag_psi, atol=1e-05).all()
                assert not torch.isclose(fa.diag_psi, old_diag_psi, atol=1e-05).all()
