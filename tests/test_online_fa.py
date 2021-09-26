import numpy as np
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
    def test_init_F_orthogonal_columns(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        Q = fa.F.t().mm(fa.F)
        assert torch.isclose(Q, torch.eye(latent_dim), atol=1e5, rtol=1).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_init_diag_psi(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert (fa.diag_psi == torch.ones(fa.observation_dim, 1)).all()

    def test_init_t(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert fa.t == 0

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_commons_t(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        fa._update_commons(torch.randn(observation_dim, 1))
        assert fa.t == 1
        fa._update_commons(torch.randn(observation_dim))
        assert fa.t == 2

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_commons_c(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa.c, theta1, atol=1e-05).all()
        theta2 = torch.randn(observation_dim, 1)
        fa._update_commons(theta2)
        assert torch.isclose(fa.c, (theta1 + theta2) / 2, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_commons_centred_observation(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa._d, torch.zeros(observation_dim), atol=1e-05).all()
        theta2 = torch.randn(observation_dim, 1)
        fa._update_commons(theta2)
        assert torch.isclose(fa._d, theta2 - (theta1 + theta2) / 2, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_commons_inv_psi(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa._diag_inv_psi, 1 / fa.diag_psi, atol=1e-05).all()
        theta2 = torch.randn(observation_dim)
        fa._update_commons(theta2)
        assert torch.isclose(fa._diag_inv_psi, 1 / fa.diag_psi, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_update_commons_m(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
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
    def test_update_commons_sigma(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        theta2 = torch.randn(observation_dim)
        fa._update_commons(theta2)
        F = fa.F
        inv_psi = torch.diag(fa._diag_inv_psi.squeeze())
        expected_sigma = torch.linalg.inv(torch.eye(latent_dim) + F.t().mm(inv_psi).mm(F))
        assert torch.isclose(fa._sigma, expected_sigma, atol=1e-05).all()

    def test_get_covariance(self):
        fa = OnlineFactorAnalysis(observation_dim=3, latent_dim=2)
        fa.F = torch.Tensor([
            [1, 2],
            [-3, 4],
            [-1, 2],
        ])
        fa.diag_psi = torch.Tensor([1, 2, 3])
        expected_covariance = torch.Tensor([
            [6, 5, 3],
            [5, 27, 11],
            [3, 11, 8],
        ])
        actual_covariance = fa.get_covariance()
        assert torch.isclose(actual_covariance, expected_covariance, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [1, 5, 8])
    @pytest.mark.parametrize('n_samples', [1, 10, 100])
    @pytest.mark.parametrize('random_seed', [1, None])
    def test_sample_shape(self, observation_dim, latent_dim, n_samples, random_seed):
        fa = OnlineFactorAnalysis(observation_dim=observation_dim, latent_dim=latent_dim)
        observations = fa.sample(n_samples, random_seed)
        assert observations.shape == (n_samples, observation_dim)

    def test_sample_mean(self):
        fa = OnlineFactorAnalysis(observation_dim=10, latent_dim=5)
        observations = fa.sample(n_samples=1000, random_seed=1)
        actual_mean = observations.mean(dim=0)
        assert torch.isclose(actual_mean, fa.get_mean(), atol=1e-1).all()

    def test_sample_covariance(self):
        fa = OnlineFactorAnalysis(observation_dim=10, latent_dim=5)
        observations = fa.sample(n_samples=10000, random_seed=1)
        actual_covar = torch.from_numpy(np.cov(observations.numpy().T)).float()
        assert torch.isclose(actual_covar, fa.get_covariance(), atol=1e-1).all()
