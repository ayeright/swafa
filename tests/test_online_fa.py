import torch
import pytest

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
        assert (fa.diag_psi == torch.ones(observation_dim, 1)).all()

    def test_init_t(self):
        observation_dim = 4
        latent_dim = 3
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        assert fa.t == 0

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    def test_invert_psi(self, observation_dim, latent_dim):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim)
        fa.diag_psi = torch.randn(observation_dim, 1)
        inv_diag_psi = fa._invert_psi()
        assert torch.isclose(inv_diag_psi, 1 / fa.diag_psi, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_t(self, observation_dim, latent_dim, init_factors_noise_std):
        outputs = self._update_commons(observation_dim, latent_dim, init_factors_noise_std)
        assert outputs['t1'] == 1
        assert outputs['t2'] == 2

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_c(self, observation_dim, latent_dim, init_factors_noise_std):
        outputs = self._update_commons(observation_dim, latent_dim, init_factors_noise_std)
        c1 = outputs['c1']
        c2 = outputs['c2']
        theta1 = outputs['theta1']
        theta2 = outputs['theta2']
        assert torch.isclose(c1, theta1, atol=1e-05).all()
        assert torch.isclose(c2, (theta1 + theta2) / 2, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_centred_observation(self, observation_dim, latent_dim, init_factors_noise_std):
        outputs = self._update_commons(observation_dim, latent_dim, init_factors_noise_std)
        d1 = outputs['d1']
        d2 = outputs['d2']
        theta1 = outputs['theta1']
        theta2 = outputs['theta2']
        assert torch.isclose(d1, torch.zeros(observation_dim), atol=1e-05).all()
        assert torch.isclose(d2, theta2 - (theta1 + theta2) / 2, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_m(self, observation_dim, latent_dim, init_factors_noise_std):
        outputs = self._update_commons(observation_dim, latent_dim, init_factors_noise_std)
        fa = outputs['fa']
        d = outputs['d2']
        m = outputs['m2']
        inv_psi = torch.diag(fa._invert_psi().squeeze())
        F = fa.F
        expected_m = torch.linalg.inv(torch.eye(latent_dim) + F.t().mm(inv_psi).mm(F)).mm(F.t()).mm(inv_psi).mm(d)
        assert torch.isclose(m, expected_m, atol=1e-05).all()

    @pytest.mark.parametrize("observation_dim", [10, 20])
    @pytest.mark.parametrize("latent_dim", [5, 8])
    @pytest.mark.parametrize("init_factors_noise_std", [1e-3, 1e-2])
    def test_update_commons_sigma(self, observation_dim, latent_dim, init_factors_noise_std):
        outputs = self._update_commons(observation_dim, latent_dim, init_factors_noise_std)
        fa = outputs['fa']
        sigma = outputs['sigma2']
        inv_psi = torch.diag(fa._invert_psi().squeeze())
        F = fa.F
        expected_sigma = torch.linalg.inv(torch.eye(latent_dim) + F.t().mm(inv_psi).mm(F))
        assert torch.isclose(sigma, expected_sigma, atol=1e-05).all()

    @staticmethod
    def _update_commons(observation_dim: int, latent_dim: int, init_factors_noise_std: float):
        fa = OnlineFactorAnalysis(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std)
        fa.diag_psi = torch.randn(observation_dim, 1)
        theta1 = torch.randn(observation_dim, 1)
        d1, diag_inv_psi1, m1, sigma1 = fa._update_commons(theta1)
        t1 = fa.t
        c1 = fa.c
        theta2 = torch.randn(observation_dim, 1)
        d2, diag_inv_psi2, m2, sigma2 = fa._update_commons(theta2)
        t2 = fa.t
        c2 = fa.c
        return dict(
            fa=fa,
            theta1=theta1,
            t1=t1,
            c1=c1,
            d1=d1,
            diag_inv_psi1=diag_inv_psi1,
            m1=m1,
            sigma1=sigma1,
            theta2=theta2,
            t2=t2,
            c2=c2,
            d2=d2,
            diag_inv_psi2=diag_inv_psi2,
            m2=m2,
            sigma2=sigma2
        )
