import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything

from swafa.models import FeedForwardNet


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
    @pytest.mark.parametrize("activation_fn", [None, nn.ReLU()])
    @pytest.mark.parametrize("n_samples", [1, 4])
    def test_forward(self, input_dim, hidden_dims, activation_fn, n_samples):
        net = FeedForwardNet(input_dim, hidden_dims, activation_fn=activation_fn)
        X = torch.rand(n_samples, input_dim)
        y = net(X)

        assert y.shape == (n_samples,)

    @pytest.mark.parametrize("input_dim", [5])
    @pytest.mark.parametrize("hidden_dims", [None, [4]])
    @pytest.mark.parametrize("activation_fn", [None, nn.ReLU()])
    @pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.BCEWithLogitsLoss()])
    @pytest.mark.parametrize("n_samples", [32, 33])
    def test_fit_with_validation(self, input_dim, hidden_dims, activation_fn, loss_fn, n_samples):
        seed_everything(42, workers=True)
        net = FeedForwardNet(input_dim, hidden_dims, activation_fn=activation_fn, loss_fn=loss_fn)
        original_weights = [torch.clone(w) for w in net.parameters()]

        trainer = Trainer(deterministic=True, max_epochs=5, check_val_every_n_epoch=1)

        train_dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, drop_last=True)

        val_dataset = TensorDataset(torch.randn(n_samples, input_dim), torch.empty(n_samples).random_(2))
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=4, drop_last=False)

        trainer.fit(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

        for w_old, w_new in zip(original_weights, net.parameters()):
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
