import pytest

from swafa.models import FeedForwardNet
from swafa.fa import OnlineGradientFactorAnalysis
from swafa.posterior import ModelPosterior


class TestModelPosterior:

    @pytest.mark.parametrize(
        "input_dim, hidden_dims, posterior_latent_dim, expected_posterior_dim",
        [
            (5, None, 3, 5 + 1),
            (6, [4], 2, (6 + 1) * 4 + (4 + 1)),
            (7, [6, 9], 5, (7 + 1) * 6 + (6 + 1) * 9 + (9 + 1)),
        ]
    )
    def test_posterior_dimension(self, input_dim, hidden_dims, posterior_latent_dim, expected_posterior_dim):
        net = FeedForwardNet(input_dim, hidden_dims)

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=posterior_latent_dim),
        )

        assert model_posterior.weight_posterior.latent_dim == posterior_latent_dim
        assert model_posterior.weight_posterior.observation_dim == expected_posterior_dim
