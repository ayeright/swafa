import torch

from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis

OnlineFactorAnalysis = OnlineGradientFactorAnalysis  # can't test OnlineFactorAnalysis directly as it is abstract


class TestOnlineFactorAnalysis:

    def test_init_c(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert (fa.c == torch.zeros(observation_dim, 1)).all()

    def test_init_F_shape(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert fa.F.shape == (observation_dim, latent_dim)

    def test_init_F_diagonal(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        x = torch.diag(fa.F)
        assert torch.isclose(x, torch.ones(latent_dim)).all()

    def test_init_F_off_diagonal(self):
        observation_dim = 4
        latent_dim = 3
        noise_std = 1e-3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=noise_std)
        for i in range(observation_dim):
            for j in range(latent_dim):
                if i != j:
                    assert abs(fa.F[i, j]) < noise_std * 10

    def test_init_diag_psi(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert (fa.diag_psi == torch.ones(observation_dim, 1)).all()

    def test_init_diag_t(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert fa.t == 0

    def test_update_commons_t(self):
        observation_dim = 3
        latent_dim = 2
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        theta = torch.randn(observation_dim, 1)
        fa._update_commons(theta)
        assert fa.t == 1
        fa._update_commons(theta)
        assert fa.t == 2

    def test_update_commons_c(self):
        observation_dim = 3
        latent_dim = 2
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        theta1 = torch.randn(observation_dim, 1)
        fa._update_commons(theta1)
        assert torch.isclose(fa.c, theta1, atol=1e-05).all()
        theta2 = torch.randn(observation_dim, 1)
        fa._update_commons(theta2)
        assert torch.isclose(fa.c, (theta1 + theta2) / 2, atol=1e-05).all()

    def test_update_commons_centred_observation(self):
        observation_dim = 3
        latent_dim = 2
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        theta1 = torch.randn(observation_dim, 1)
        d1, _, _, _ = fa._update_commons(theta1)
        assert torch.isclose(d1, torch.zeros(observation_dim), atol=1e-05).all()
        theta2 = torch.randn(observation_dim, 1)
        d2, _, _, _ = fa._update_commons(theta2)
        assert torch.isclose(d2, theta2 - (theta1 + theta2) / 2, atol=1e-05).all()
