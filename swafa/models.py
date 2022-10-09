from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer, Adam
from pytorch_lightning import LightningModule


class FeedForwardNet(LightningModule):
    """
    A feed forward neural network with a single output.

    Implements functionality which allows it to be used with a PyTorch Lightning Trainer.

    Args:
        input_dim: The input dimension of the neural network.
        hidden_dims: The dimension of each hidden layer in the neural network. hidden_dims[i] is the dimension of the
            i-th hidden layer. If None, the input will be connected directly to the output.
        hidden_activation_fn: The activation function to apply to the output of each hidden layer. If None, will be set
            to the identity activation function.
        output_activation_fn: The activation function to apply to the final output when predicting. When training,
            validating or testing, no activation will be applied to the output. Hence, the loss function should take the
            un-activated outputs as input. If None, will be set to the identity activation function.
        bias: Whether or not to include a bias term in the linear layers.
        optimiser_class: The class of the PyTorch optimiser to use for training the neural network.
        optimiser_kwargs: Keyword arguments for the optimiser class.
        loss_fn: The PyTorch loss function to use for training the model. Will be applied to the un-activated outputs
            of the neural network.
        loss_multiplier: A constant with which to multiply the loss of each batch. Useful if an estimate of the total
            loss over the full dataset is needed.
        random_seed: The random seed for initialising the weights of the neural network. If None, won't be reproducible.

    Attributes:
        hidden_layers: (torch.nn.ModuleList) A list of torch.nn.Linear, corresponding to dimensions specified in
            hidden_dims.
        output_layer: (torch.nn.Linear) A linear layer with a single output.
    """

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None,
                 hidden_activation_fn: Optional[nn.Module] = None, output_activation_fn: Optional[nn.Module] = None,
                 bias: bool = True, optimiser_class: Optimizer = Adam, optimiser_kwargs: Optional[dict] = None,
                 loss_fn: nn.Module = nn.MSELoss(), loss_multiplier: float = 1.0, random_seed: Optional[int] = None):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or []
        self.hidden_activation_fn = hidden_activation_fn or self._identity_fn
        self.output_activation_fn = output_activation_fn or self._identity_fn
        self.optimiser_class = optimiser_class
        self.optimiser_kwargs = optimiser_kwargs or dict(lr=1e-3)
        self.loss_fn = loss_fn
        self.loss_multiplier = loss_multiplier

        self.hidden_layers = nn.ModuleList()
        d_in = deepcopy(input_dim)
        for d_out in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(d_in, d_out, bias=bias))
            d_in = d_out

        self.output_layer = nn.Linear(d_in, 1, bias=bias)

    @staticmethod
    def _identity_fn(X: Tensor) -> Tensor:
        """
        An function which returns the input unchanged.

        Args:
            X: A Tensor of any shape.

        Returns:
            Exactly the same as the unput.
        """
        return X

    def forward(self, X: Tensor, activate_output: bool = False) -> Tensor:
        """
        Run the forward pass of the neural network.

        Args:
            X: Input features. Of shape (n_samples, n_features).
            activate_output: Whether or not to apply the activation function to the outputs.

        Returns:
            Neural network outputs. Of shape (n_samples,).
        """
        for layer in self.hidden_layers:
            X = self.hidden_activation_fn(layer(X))

        y_hat = self.output_layer(X).squeeze(dim=1)
        if activate_output:
            return self.output_activation_fn(y_hat)
        return y_hat

    def configure_optimizers(self) -> Optimizer:
        """
        Initialise the optimiser which will be used to train the neural network.

        Returns:
            The initialised optimiser
        """
        return self.optimiser_class(self.parameters(), **self.optimiser_kwargs)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the training loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch training loss. Of shape (1,).
        """
        return self._step(batch, batch_idx)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the validation loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch validation loss. Of shape (1,).
        """
        return self._step(batch, batch_idx)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the test loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch test loss. Of shape (1,).
        """
        return self._step(batch, batch_idx)

    def _step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch loss. Of shape (1,).
        """
        X, y = batch
        y_hat = self(X)
        return self.loss_fn(y_hat, y) * self.loss_multiplier

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        """
        Predict the outputs for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.
            dataloader_idx: The index of the dataloader (may be more than one) from which the batch was sampled.

        Returns:
            The activated outputs. Of shape (batch_size,).
        """
        return self(batch[0], activate_output=True)

    def validation_epoch_end(self, step_losses: List[Tensor]) -> Dict[str, Tensor]:
        """
        Compute the average validation loss over all batches.

        Log the loss under the name 'epoch_val_loss'.

        Args:
            step_losses: The validation loss for each individual batch. Each one of shape (1,).

        Returns:
            A dict of the form {'epoch_val_loss': loss}, where loss is the average validation loss, of shape (1,).
        """
        loss = self._average_loss(step_losses)
        metrics = dict(epoch_val_loss=loss)
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, step_losses: List[Tensor]) -> Dict[str, Tensor]:
        """
        Compute the average test loss over all batches.

        Log the loss under the name 'epoch_test_loss'.

        Args:
            step_losses: The test loss for each individual batch. Each one of shape (1,).

        Returns:
            A dict of the form {'epoch_test_loss': loss}, where loss is the average test loss, of shape (1,).
        """
        loss = self._average_loss(step_losses)
        metrics = dict(epoch_test_loss=loss)
        self.log_dict(metrics)
        return metrics

    @staticmethod
    def _average_loss(step_losses: List[Tensor]) -> Tensor:
        """
        Compute the average of all losses.

        Args:
            step_losses: Individual losses. Each one of shape (1,).

        Returns:
            The average loss. Of shape (1,).
        """
        return torch.stack(step_losses).mean()


class FeedForwardGaussianNet(FeedForwardNet):
    """
    A feed forward neural network which predicts the parameters of a 1D Gaussian distribution for each input.

    Implements functionality which allows it to be used with a PyTorch Lightning Trainer.

    Args:
        input_dim: The input dimension of the neural network.
        hidden_dims: The dimension of each hidden layer in the neural network. hidden_dims[i] is the dimension of the
            i-th hidden layer. If None, the input will be connected directly to the output.
        hidden_activation_fn: The activation function to apply to the output of each hidden layer. If None, will be set
            to the identity activation function.
        bias: Whether or not to include a bias term in the linear layers.
        optimiser_class: The class of the PyTorch optimiser to use for training the neural network.
        optimiser_kwargs: Keyword arguments for the optimiser class.
        loss_multiplier: A constant with which to multiply the average loss of each batch. Useful if an estimate of the
            total loss over the full dataset is needed.
        target_variance: A constant variance to use for each Gaussian. If None, an extra output layer will be added to
            the network to predict the variance as well as the mean.
        variance_epsilon: Value used to clamp the variance for numerical stability.
        random_seed: The random seed for initialising the weights of the neural network. If None, won't be reproducible.

    Attributes:
        hidden_layers: (torch.nn.ModuleList) A list of torch.nn.Linear, corresponding to dimensions specified in
            hidden_dims.
        output_layer: (torch.nn.Linear) A linear layer with a single output which predicts the mean of the Gaussian.
        log_variance_layer: (Optional[torch.nn.Linear]) A linear layer with a single output which predicts the log of
            the variance of the Gaussian. If target_variance is not None, will also be None.
    """

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None,
                 hidden_activation_fn: Optional[nn.Module] = None, bias: bool = True, optimiser_class: Optimizer = Adam,
                 optimiser_kwargs: Optional[dict] = None, loss_multiplier: float = 1.0, target_variance: float = None,
                 variance_epsilon: float = 1e-6, random_seed: Optional[int] = None):

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            hidden_activation_fn=hidden_activation_fn,
            bias=bias,
            optimiser_class=optimiser_class,
            optimiser_kwargs=optimiser_kwargs,
            loss_fn=nn.GaussianNLLLoss(reduction='mean', eps=variance_epsilon),
            loss_multiplier=loss_multiplier,
            random_seed=random_seed,
        )

        self.log_variance_layer = None
        if target_variance is None:
            d_in = self.output_layer.in_features
            self.log_variance_layer = nn.Linear(d_in, 1, bias=bias)

        self.target_variance = target_variance

    def forward(self, X: Tensor) -> (Tensor, Tensor):
        """
        Run the forward pass of the neural network.

        Args:
            X: Input features. Of shape (n_samples, n_features).

        Returns:
            mu: Predicted mean of each input. Of shape (n_samples,).
            var: Predicted variance of each input. Of shape (n_samples,).
        """
        for layer in self.hidden_layers:
            X = self.hidden_activation_fn(layer(X))

        mu = self.output_layer(X).squeeze(dim=1)

        if self.log_variance_layer is None:
            var = torch.ones_like(mu) * self.target_variance
        else:
            var = torch.exp(self.log_variance_layer(X).squeeze(dim=1))

        return mu, var

    def _step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the Gaussian negative log likelihood loss for a single batch of data.

        The average batch loss is multiplied by self._loss_multiplier.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch loss. Of shape (1,).
        """
        X, y = batch

        print(X)

        mu, var = self(X)

        return self.loss_fn(mu, y, var) * self.loss_multiplier
