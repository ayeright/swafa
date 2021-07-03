import pytest
import torch

from swafa.fa import OnlineGradientFactorAnalysis

OnlineFactorAnalysis = OnlineGradientFactorAnalysis  # can't test OnlineFactorAnalysis directly as it is abstract


class TestOnlineFactorAnalysis:

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_init_c(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert (fa.c == torch.zeros(observation_dim, 1)).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_init_F_shape(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert fa.F.shape == (observation_dim, latent_dim)

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_init_F_diagonal(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        x = torch.diag(fa.F)
        assert torch.isclose(x, torch.ones(latent_dim)).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_init_F_off_diagonal(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        for i in range(observation_dim):
            for j in range(latent_dim):
                if i != j:
                    assert abs(fa.F[i, j]) < init_factors_noise_std * 10

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_init_diag_psi(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        max_diag_FFt = torch.diag(fa.F.mm(fa.F.t())).max()
        expected_diag_psi = max_diag_FFt * torch.ones(fa.observation_dim, 1)
        assert (fa.diag_psi == expected_diag_psi).all()

    def test_init_t(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert fa.t == 0

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_t(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        fa._update_commons(torch.randn(observation_dim, 1))
        assert fa.t == 1
        fa._update_commons(torch.randn(observation_dim))
        assert fa.t == 2

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_c(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa.c, theta1, atol=1e-05).all()
        theta2 = torch.randn(observation_dim, 1)
        fa._update_commons(theta2)
        assert torch.isclose(fa.c, (theta1 + theta2) / 2, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_centred_observation(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa._d, torch.zeros(observation_dim), atol=1e-05).all()
        theta2 = torch.randn(observation_dim, 1)
        fa._update_commons(theta2)
        assert torch.isclose(fa._d, theta2 - (theta1 + theta2) / 2, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_inv_psi(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa._diag_inv_psi, 1 / fa.diag_psi, atol=1e-05).all()
        theta2 = torch.randn(observation_dim)
        fa._update_commons(theta2)
        assert torch.isclose(fa._diag_inv_psi, 1 / fa.diag_psi, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_m(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa._m, torch.zeros(latent_dim), atol=1e-05).all()
        theta2 = torch.randn(observation_dim)
        fa._update_commons(theta2)
        F = fa.F
        inv_psi = torch.diag(fa._diag_inv_psi.squeeze())
        expected_m = torch.linalg.inv(torch.eye(latent_dim) + F.t().mm(inv_psi).mm(F)).mm(F.t()).mm(inv_psi).mm(fa._d)
        assert torch.isclose(fa._m, expected_m, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_sigma(self, observation_dim, latent_dim, init_factors_noise_std):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        theta2 = torch.randn(observation_dim)
        fa._update_commons(theta2)
        F = fa.F
        inv_psi = torch.diag(fa._diag_inv_psi.squeeze())
        expected_sigma = torch.linalg.inv(torch.eye(latent_dim) + F.t().mm(inv_psi).mm(F))
        assert torch.isclose(fa._sigma, expected_sigma, atol=1e-05).all()
