import torch
import pytest

from swafa.fa import OnlineEMFactorAnalysis


class TestOnlineEMFactorAnalysis:

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_B_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        m_times_mt = []
        for _ in range(10):
            _, _, m, _ = fa._update_commons(torch.randn(observation_dim))
            m_times_mt.append(m.mm(m.t()))
            fa._update_B_hat(m)

        expected_B_hat = torch.dstack(m_times_mt).mean(dim=2)
        assert torch.isclose(fa.B_hat, expected_B_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_calc_H_hat_updates_B_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        m_times_mt = []
        for _ in range(10):
            _, _, m, sigma = fa._update_commons(torch.randn(observation_dim))
            m_times_mt.append(m.mm(m.t()))
            fa._calc_H_hat(m, sigma)

        expected_B_hat = torch.dstack(m_times_mt).mean(dim=2)
        assert torch.isclose(fa.B_hat, expected_B_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_calc_H_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        for _ in range(10):
            _, _, m, sigma = fa._update_commons(torch.randn(observation_dim))
            actual_H_hat = fa._calc_H_hat(m, sigma)
            expected_H_hat = sigma + fa.B_hat
            assert torch.isclose(actual_H_hat, expected_H_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_A_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        d_times_mt = []
        for _ in range(10):
            d, _, m, _ = fa._update_commons(torch.randn(observation_dim))
            d_times_mt.append(d.mm(m.t()))
            fa._update_A_hat(d, m)

        expected_A_hat = torch.dstack(d_times_mt).mean(dim=2)
        assert torch.isclose(fa.A_hat, expected_A_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_F(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        for i in range(10):
            d, _, m, sigma = fa._update_commons(torch.randn(observation_dim))
            H_hat = fa._calc_H_hat(m, sigma)
            fa._update_A_hat(d, m)
            if i > 0:
                fa._update_F(H_hat)
                fa._update_psi(d, H_hat)
                expected_F = fa.A_hat.mm(torch.linalg.inv(H_hat))
                assert torch.isclose(fa.F, expected_F, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_d_squared_hat(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        d_squared = []
        for _ in range(10):
            d, _, _, _ = fa._update_commons(torch.randn(observation_dim))
            d_squared.append(d ** 2)
            fa._update_d_squared_hat(d)

        expected_d_squared_hat = torch.dstack(d_squared).mean(dim=2)
        assert torch.isclose(fa.d_squared_hat, expected_d_squared_hat, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_psi(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineEMFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        d_times_dt = []
        for i in range(10):
            d, diag_inv_psi, m, sigma = fa._update_commons(torch.randn(observation_dim))
            d_times_dt.append(d.mm(d.t()))
            H_hat = fa._calc_H_hat(m, sigma)
            fa._update_A_hat(d, m)
            if i > 0:
                fa._update_F(H_hat)
                fa._update_psi(d, H_hat)

                expected_diag_psi = torch.diag(
                    torch.dstack(d_times_dt).mean(dim=2)
                    - 2 * fa.F.mm(fa.A_hat.t())
                    + fa.F.mm(H_hat).mm(fa.F.t())
                ).reshape(-1, 1)

                assert torch.isclose(fa.diag_psi, expected_diag_psi, atol=1e-05).all()
