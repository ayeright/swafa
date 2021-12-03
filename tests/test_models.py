import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything

from swafa.models import FeedForwardNet, FeedForwardGaussianNet


class TestFeedForwardNet:

    @pytest.mark.parametrize("input_dim", [3, 5])
    @pytest.mark.parametrize("hidden_dims", [None, [4], [8, 4]])
    def test_init_layers(self, input_dim, hidden_dims):
        net = FeedForwardNet(input_dim, hidden_dims, random_seed=42)
        hidden_dims = hidden_dims or []

        assert len(net.hidden_layers) == len(hidden_dims)

        d_in = input_dim
        for i, layer in enumerate(net.hidden_layers):
            d_out = hidden_dims[i]
            assert layer.in_features == d_in
            assert layer.out_features == d_out
            d_in = d_out

        assert net.output_layer.in_features == d_in
        assert net.output_layer.out_features == 1

    @pytest.mark.parametrize("input_dim", [3, 5])
    @pytest.mark.parametrize("hidden_dims", [None, [4], [8, 4]])
    @pytest.mark.parametrize("hidden_activation_fn", [None, nn.ReLU()])
    @pytest.mark.parametrize("n_samples", [1, 4])
    @pytest.mark.parametrize("activate_output", [True, False])
    def test_forward(self, input_dim, hidden_dims, hidden_activation_fn, n_samples, activate_output):

        def zero_activation_fn(x):
            return x * 0

        net = FeedForwardNet(input_dim, hidden_dims, hidden_activation_fn=hidden_activation_fn,
                             output_activation_fn=zero_activation_fn)

        X = torch.rand(n_samples, input_dim)
        y = net(X, activate_output=activate_output)

        assert y.shape == (n_samples,)
        assert (y == 0).all() == activate_output

    @pytest.mark.parametrize("input_dim", [5])
    @pytest.mark.parametrize("hidden_dims", [None, [4]])
    @pytest.mark.parametrize("hidden_activation_fn", [None, nn.ReLU()])
    @pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.BCEWithLogitsLoss()])
    @pytest.mark.parametrize("n_samples", [32, 33])
    def test_fit_with_validation(self, input_dim, hidden_dims, hidden_activation_fn, loss_fn, n_samples):
        seed_everything(42, workers=True)
        net = FeedForwardNet(input_dim, hidden_dims, hidden_activation_fn=hidden_activation_fn, loss_fn=loss_fn)
        original_weights = [torch.clone(w) for w in net.parameters()]

        trainer = Trainer(deterministic=True, max_epochs=5, check_val_every_n_epoch=1)

        train_dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, drop_last=True)

        val_dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=4, drop_last=False)

        trainer.fit(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

        for w_old, w_new in zip(original_weights, net.parameters()):
            assert not torch.isnan(w_new).any()
            assert not torch.isclose(w_old, w_new).all()

    @pytest.mark.parametrize("input_dim", [5])
    @pytest.mark.parametrize("hidden_dims", [None, [4]])
    @pytest.mark.parametrize("n_samples", [32, 33])
    def test_validate(self, input_dim, hidden_dims, n_samples):
        net = FeedForwardNet(input_dim, hidden_dims)
        trainer = Trainer()
        val_dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=4, drop_last=False)

        result = trainer.validate(net, val_dataloaders=val_dataloader)

        assert len(result) == 1
        assert list(result[0].keys()) == ['epoch_val_loss']

    @pytest.mark.parametrize("input_dim", [5])
    @pytest.mark.parametrize("hidden_dims", [None, [4]])
    @pytest.mark.parametrize("n_samples", [32, 33])
    def test_test(self, input_dim, hidden_dims, n_samples):
        net = FeedForwardNet(input_dim, hidden_dims)
        trainer = Trainer()
        test_dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4, drop_last=False)

        result = trainer.test(net, test_dataloaders=test_dataloader)

        assert len(result) == 1
        assert list(result[0].keys()) == ['epoch_test_loss']

    @pytest.mark.parametrize("input_dim", [5])
    @pytest.mark.parametrize("hidden_dims", [None, [4]])
    @pytest.mark.parametrize("n_samples", [32, 33])
    def test_predict(self, input_dim, hidden_dims, n_samples):
        net = FeedForwardNet(input_dim, hidden_dims)
        trainer = Trainer()
        dataset = TensorDataset(torch.randn(n_samples, input_dim))
        dataloader = DataLoader(dataset, shuffle=False, batch_size=4, drop_last=False)

        result = trainer.predict(net, dataloaders=dataloader)

        assert len(torch.cat(result)) == n_samples

    @pytest.mark.parametrize("input_dim", [5])
    @pytest.mark.parametrize("hidden_dims", [None, [4]])
    @pytest.mark.parametrize("n_samples", [32, 33])
    def test_predict_with_sigmoid_activation(self, input_dim, hidden_dims, n_samples):
        net = FeedForwardNet(input_dim, hidden_dims, output_activation_fn=torch.sigmoid)
        trainer = Trainer()
        dataset = TensorDataset(torch.randn(n_samples, input_dim))
        dataloader = DataLoader(dataset, shuffle=False, batch_size=4, drop_last=False)

        result = trainer.predict(net, dataloaders=dataloader)

        for batch_predictions in result:
            assert ((batch_predictions >= 0) & (batch_predictions <= 1)).all()


class TestFeedForwardGaussianNet:

    @pytest.mark.parametrize("input_dim", [3, 5])
    @pytest.mark.parametrize("hidden_dims", [None, [4], [8, 4]])
    @pytest.mark.parametrize("hidden_activation_fn", [None, nn.ReLU()])
    @pytest.mark.parametrize("n_samples", [1, 4])
    @pytest.mark.parametrize("target_variance", [None, 2])
    def test_forward(self, input_dim, hidden_dims, hidden_activation_fn, n_samples, target_variance):
        net = FeedForwardGaussianNet(
            input_dim, hidden_dims, hidden_activation_fn=hidden_activation_fn, target_variance=target_variance,
        )

        X = torch.rand(n_samples, input_dim)
        mu, var = net(X)

        assert mu.shape == (n_samples,)
        assert var.shape == (n_samples,)
        assert (var > 0).all()

        if target_variance is not None:
            assert (var == target_variance).all()

    @pytest.mark.parametrize("input_dim", [3, 5])
    @pytest.mark.parametrize("hidden_dims", [None, [4], [8, 4]])
    @pytest.mark.parametrize("n_samples", [1, 4])
    @pytest.mark.parametrize("loss_multiplier", [1, 5])
    def test_step(self, input_dim, hidden_dims, n_samples, loss_multiplier):
        net = FeedForwardGaussianNet(input_dim, hidden_dims, loss_multiplier=loss_multiplier)

        X = torch.rand(n_samples, input_dim)
        y = torch.randn(n_samples)
        batch = (X, y)
        actual_loss = net._step(batch, batch_idx=1)

        mu, var = net(X)
        expected_loss = net.loss_fn(mu, y, var) * loss_multiplier

        assert torch.isclose(actual_loss, expected_loss)

    @pytest.mark.parametrize("input_dim", [5])
    @pytest.mark.parametrize("hidden_dims", [None, [4]])
    @pytest.mark.parametrize("hidden_activation_fn", [None, nn.ReLU()])
    @pytest.mark.parametrize("loss_multiplier", [1.0, 2.0])
    @pytest.mark.parametrize("target_variance", [None, 1.0])
    @pytest.mark.parametrize("variance_epsilon", [1e-1, 1e-6])
    @pytest.mark.parametrize("n_samples", [32, 33])
    def test_fit(self, input_dim, hidden_dims, hidden_activation_fn, loss_multiplier, target_variance, variance_epsilon,
                 n_samples):
        seed_everything(42, workers=True)
        net = FeedForwardGaussianNet(
            input_dim, hidden_dims, hidden_activation_fn=hidden_activation_fn, loss_multiplier=loss_multiplier,
            target_variance=target_variance, variance_epsilon=variance_epsilon,
        )
        original_weights = [torch.clone(w) for w in net.parameters()]

        trainer = Trainer(deterministic=True, max_epochs=5)

        train_dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, drop_last=True)

        trainer.fit(net, train_dataloader=train_dataloader)

        for w_old, w_new in zip(original_weights, net.parameters()):
            assert not torch.isnan(w_new).any()
            assert not torch.isclose(w_old, w_new).all()
