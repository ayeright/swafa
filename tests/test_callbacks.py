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
        "n_samples, batch_size, n_epochs, update_epoch_start, iterate_averaging_window_size, expected_n_updates",
        [
            (32, 4, 5, 1, 1, int(32 / 4) * 5),
            (32, 4, 5, 3, 1, int(32 / 4) * (5 - 2)),
            (32, 4, 8, 0.5, 1, int(32 / 4) * (8 - 3)),
            (32, 4, 9, 0.5, 1, int(32 / 4) * (9 - 3)),
            (32, 4, 5, 1, 2, (int(32 / 4) * 5 / 2)),
            (32, 4, 5, 3, 2, (int(32 / 4) * (5 - 2) / 2)),
            (32, 4, 8, 0.5, 2, (int(32 / 4) * (8 - 3) / 2)),
            (32, 4, 9, 0.5, 2, (int(32 / 4) * (9 - 3) / 2)),
        ]
    )
    def test_posterior_updates(self, n_samples, batch_size, n_epochs, update_epoch_start, iterate_averaging_window_size,
                               expected_n_updates):
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
            iterate_averaging_window_size=iterate_averaging_window_size,
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

    def test_raises_error_if_model_and_posterior_do_not_match(self):
        n_samples = 32
        input_dim = 10
        net = FeedForwardNet(input_dim=input_dim, hidden_dims=[10])

        model_posterior = ModelPosterior(
            model=FeedForwardNet(input_dim=input_dim, hidden_dims=[5]),
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        callback = WeightPosteriorCallback(
            posterior=model_posterior.weight_posterior,
            update_epoch_start=1,
        )

        trainer = Trainer(max_epochs=10, callbacks=[callback])

        dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        dataloader = DataLoader(dataset, batch_size=4, drop_last=True)

        with pytest.raises(RuntimeError):
            trainer.fit(net, train_dataloader=dataloader)

    def test_update_weight_window_average(self):
        input_dim = 10
        net = FeedForwardNet(input_dim=input_dim, bias=False)

        model_posterior = ModelPosterior(
            model=net,
            weight_posterior_class=OnlineGradientFactorAnalysis,
            weight_posterior_kwargs=dict(latent_dim=3),
        )

        callback = WeightPosteriorCallback(
            posterior=model_posterior.weight_posterior,
            update_epoch_start=1,
        )

        weights1 = torch.randn(input_dim)
        callback._weight_window_average = torch.clone(weights1)
        callback._window_index = 1
        weights2 = torch.randn(input_dim)
        callback._update_weight_window_average(weights2)

        assert torch.isclose(callback._weight_window_average, (weights1 + weights2) / 2).all()
        assert callback._window_index == 2
