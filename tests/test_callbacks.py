import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer

from swafa.models import FeedForwardNet
from swafa.callbacks import WeightPosteriorCallback
from swafa.fa import OnlineGradientFactorAnalysis
from swafa.posterior import ModelPosterior


class TestWeightPosteriorUpdate:

    @pytest.mark.parametrize(
        "input_dim, hidden_dims, expected_n_weights",
        [
            (5, None, 5 + 1),
            (6, [4], (6 + 1) * 4 + (4 + 1)),
            (7, [6, 9], (7 + 1) * 6 + (6 + 1) * 9 + (9 + 1)),
        ]
    )
    def test_vectorise_weights(self, input_dim, hidden_dims, expected_n_weights):
        net = FeedForwardNet(input_dim, hidden_dims)

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        callback = WeightPosteriorCallback(posterior=model_posterior.weight_posterior)
        weights = callback._vectorise_weights(net)

        assert len(weights) == expected_n_weights

    @pytest.mark.parametrize(
        "n_samples, batch_size, n_epochs, update_epoch_start, expected_n_updates",
        [
            (32, 4, 5, 1, int(32 / 4) * 5),
            (32, 4, 5, 3, int(32 / 4) * (5 - 2)),
            (32, 4, 8, 0.5, int(32 / 4) * (8 - 3)),
            (32, 4, 9, 0.5, int(32 / 4) * (9 - 3)),
        ]
    )
    def test_posterior_updates(self, n_samples, batch_size, n_epochs, update_epoch_start, expected_n_updates):
        input_dim = 4
        hidden_dims = [8, 8]
        net = FeedForwardNet(input_dim, hidden_dims)

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        callback = WeightPosteriorCallback(
            posterior=model_posterior.weight_posterior,
            update_epoch_start=update_epoch_start,
        )

        trainer = Trainer(max_epochs=n_epochs, callbacks=[callback])

        dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        dataloader = DataLoader(dataset, batch_size=4, drop_last=True)

        trainer.fit(model_posterior.model, train_dataloader=dataloader)

        assert model_posterior.weight_posterior.t == expected_n_updates

    @pytest.mark.parametrize("bad_update_epoch_start", [0, -0.1, 1.1])
    def test_init_raises_error_for_bad_update_epoch_start(self, bad_update_epoch_start):
        net = FeedForwardNet(input_dim=10, hidden_dims=[10])

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        with pytest.raises(ValueError):
            WeightPosteriorCallback(
                posterior=model_posterior.weight_posterior,
                update_epoch_start=bad_update_epoch_start,
            )