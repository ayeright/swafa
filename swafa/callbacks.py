from typing import Any, Optional, Union

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from torch.optim import Optimizer, SGD
from torch.autograd import Variable

from swafa.custom_types import POSTERIOR_TYPE
from swafa.utils import (
    get_callback_epoch_range,
    vectorise_weights,
    vectorise_gradients,
    get_weight_dimension,
    set_weights,
    normalise_gradient,
)
from swafa.fa import OnlineGradientFactorAnalysis


class WeightPosteriorCallback(Callback):
    """
    A callback which can be used with a PyTorch Lightning Trainer to update the posterior distribution of a model's
    weights.

    The updates are performed using the weight iterates sampled after each mini-batch update. Each iterate can update
    the posterior separately, or alternatively, the update can be made using the average of the iterates within a fixed
    window.

    When this callback is used while training a model, the dimension of the posterior distribution must match the
    dimension of the model's weight space.

    Args:
        posterior: Posterior distribution over the weights of a PyTorch Lighting model.
        update_epoch_start: The training epoch on which to start updating the posterior. Integer indexing starts from 1.
            Can also specify a float between 0 and 1, which corresponds to the fraction of total epochs which should
            pass before starting to update the posterior.
        iterate_averaging_window_size: The size of the window for averaging weight iterates. An update will be made to
            the posterior using each window average. Setting this to 1 is equivalent to using each iterate to update
            the posterior separately.

    Attributes:
        first_update_epoch: The epoch on which the updates to the posterior will start.
        last_update_epoch: The epoch on which the updates to the posterior will end.
    """

    def __init__(self, posterior: POSTERIOR_TYPE, update_epoch_start: Union[int, float] = 1,
                 iterate_averaging_window_size: int = 1):
        error_msg = f"update_epoch_start should be a positive integer or a float between 0 and 1, " \
                    f"not {update_epoch_start}"
        if isinstance(update_epoch_start, int) and update_epoch_start < 1:
            raise ValueError(error_msg)
        if isinstance(update_epoch_start, float) and not (0 <= update_epoch_start <= 1):
            raise ValueError(error_msg)

        self.posterior = posterior
        self._update_epoch_start = update_epoch_start

        self.iterate_averaging_window_size = iterate_averaging_window_size
        self._weight_window_average = None
        self._window_index = 0

        self.first_update_epoch = None
        self.last_update_epoch = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when fit begins.

        Initialise the range of epochs on which the posterior will be updated and check that the dimension of the
        posterior distribution matches the dimension of the model's weight space.

        Also, initialise the average weight vector within the current window.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        self.first_update_epoch, self.last_update_epoch = get_callback_epoch_range(
            trainer, epoch_start=self._update_epoch_start,
        )
        weights = self._check_weight_dimension(pl_module)
        self._reset_weight_window_average(weights)

    def _check_weight_dimension(self, pl_module: LightningModule) -> Tensor:
        """
        Check that the dimension of the posterior distribution matches the dimension of the model's weight space.

        If not, raise a RuntimeError.

        Args:
            pl_module: The model being trained.

        Returns:
            The vectorised model weights, of shape (n_weights,).
        """
        weights = vectorise_weights(pl_module)
        weight_dim = len(weights)
        if weight_dim != self.posterior.observation_dim:
            raise RuntimeError(f"The dimension of the model and the posterior weight distribution must match, but they "
                               f"are {weight_dim} and {self.posterior.observation_dim}, respectively")

        return weights

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int, dataloader_idx: int):
        """
        Called when the train batch ends.

        If within the update epoch range, update the weight iterates window average using the latest setting of the
        model's weights.

        If the weight iterate averaging window size has been reached, use the window average to update the posterior
        distribution.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
            outputs: Not used.
            batch: Not used.
            batch_idx: Not used.
            dataloader_idx: Not used.
        """
        if self.first_update_epoch <= trainer.current_epoch <= self.last_update_epoch:
            weights = vectorise_weights(pl_module)
            self._update_weight_window_average(weights)

            if self._window_index == self.iterate_averaging_window_size:
                self.posterior.update(self._weight_window_average)
                self._reset_weight_window_average(weights)

    def _update_weight_window_average(self, weights: Tensor):
        """
        Increment window index by 1 and update the running average of the window weight iterates.

        Args:
            weights: The vectorised model weights, of shape (n_weights,).
        """
        self._window_index += 1
        self._weight_window_average = \
            self._weight_window_average + (weights - self._weight_window_average) / self._window_index

    def _reset_weight_window_average(self, weights: Tensor):
        """
        Reset the window average of the weight iterates to a tensor of 0s and reset the window index to 0.

        Args:
            weights: The vectorised model weights, of shape (n_weights,).
        """
        self._window_index = 0
        self._weight_window_average = torch.zeros_like(weights)


class FactorAnalysisVariationalInferenceCallback(Callback):
    """
    A callback which can be used with a PyTorch Lightning Trainer to learn the parameters of a factor analysis
    variational distribution of a model's weights.

    The parameters are updated to minimise the Kullback-Leibler divergence between the variational distribution and the
    true posterior of the model's weights. This is done via stochastic gradient descent.

    See [1] for full details of the algorithm.

    Args:
        latent_dim: The latent dimension of the factor analysis model used as the variational distribution.
        precision: The precision of the prior of the true posterior.
        n_gradients_per_update: The number of mini-batch gradients to use to form the expectation of the true gradient
            for each parameter update.
        optimiser_class: The class of the optimiser to use for gradient updates.
        optimiser_kwargs: Keyword arguments for the optimiser. If not given, will default to dict(lr=1e-3).
        max_grad_norm: Optional maximum norm for gradients which are used to update the parameters of the variational
            distribution.
        device: The device (CPU or GPU) on which to perform the computation. If None, uses the device for the default
            tensor type.
        random_seed: The random seed for reproducibility.

    Attributes:
        weight_dim: An integer specifying the total number of weights in the model. Note that this is computed when the
            model is fit for the first time.
        c: The bias term of the factor analysis variational distribution. A Tensor of shape (weight_dim, 1).
        F: The factor loading matrix of the factor analysis variational distribution. A Tensor of shape
            (weight_dim, latent_dim).
        diag_psi: The diagonal entries of the Gaussian noise covariance matrix of the factor analysis variational
            distribution. A Tensor of shape (weight_dim, 1).

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """

    def __init__(self, latent_dim: int, precision: float, n_gradients_per_update: int = 1,
                 optimiser_class: Optimizer = SGD, optimiser_kwargs: Optional[dict] = None,
                 max_grad_norm: Optional[float] = None, device: Optional[torch.device] = None,
                 random_seed: Optional[int] = None):
        self.latent_dim = latent_dim
        self.precision = precision
        self.n_gradients_per_update = n_gradients_per_update
        self.optimiser_class = optimiser_class
        self.optimiser_kwargs = optimiser_kwargs or dict(lr=1e-3)
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.random_seed = random_seed

        self.weight_dim = None
        self.c = None
        self.F = None
        self.diag_psi = None

        self._log_diag_psi = None
        self._h = None
        self._z = None
        self._sqrt_diag_psi_dot_z = None
        self._A = None
        self._B = None
        self._C = None
        self._sigma = None
        self._diag_inv_psi = None
        self._I = torch.eye(latent_dim, device=device)
        self._s = None
        self._var_grad_wrt_F = None
        self._var_grad_wrt_log_diag_psi = None
        self._prior_grad_wrt_c = None
        self._prior_grad_wrt_F = None
        self._prior_grad_wrt_log_diag_psi = None

        self._optimiser = None
        self._batch_counter = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when fit begins.

        If parameters of variational distribution have not already been initialised, initialise them and the optimiser
        which will update them.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        if self.weight_dim is None:
            self.weight_dim = get_weight_dimension(pl_module)
            self._init_variational_params()
            self._update_expected_gradients()
            self._init_optimiser()

    def on_batch_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when the training batch begins.

        Sample weight vector from the variational distribution and use it to set the weights of the neural network.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        weights = self._sample_weight_vector()
        set_weights(pl_module, weights)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called after loss.backward() and before optimisers are stepped.

        Use the back propagated gradient of the network's loss wrt the network's weights to compute the gradient wrt
        the parameters of the variational distribution. Accumulate these gradients.

        Periodically, use the accumulated gradients to approximate the expected gradients and update the parameters of
        the variational distribution.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        grad_weights = vectorise_gradients(pl_module)[:, None]
        self._accumulate_gradients(grad_weights)

        self._batch_counter += 1
        if self._batch_counter % self.n_gradients_per_update == 0:
            self._update_variational_params()
            self._update_expected_gradients()

    def _init_variational_params(self):
        """
        Initialise the parameters of the factor analysis variational distribution.
        """
        fa = OnlineGradientFactorAnalysis(
            observation_dim=self.weight_dim,
            latent_dim=self.latent_dim,
            device=self.device,
            random_seed=self.random_seed,
        )

        self.c = Variable(fa.c.data, requires_grad=False)  # we will compute our own gradients
        self.F = Variable(fa.F.data, requires_grad=False)
        self.diag_psi = fa.diag_psi
        self._log_diag_psi = Variable(torch.log(self.diag_psi), requires_grad=False)

        self.c.grad = torch.zeros_like(self.c.data)
        self.F.grad = torch.zeros_like(self.F.data)
        self._log_diag_psi.grad = torch.zeros_like(self._log_diag_psi.data)

    def _init_optimiser(self):
        """
        Initialise the optimiser which will be used to update the parameters of the variational distribution.
        """
        self._optimiser = self.optimiser_class([self.c, self.F, self._log_diag_psi], **self.optimiser_kwargs)

    def _sample_weight_vector(self) -> Tensor:
        """
        Generate a single sample of the neural network's weight vector from the variational distribution.

        Returns:
            Sample of shape (self.weight_dim,).
        """
        self._h = torch.normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))[:, None]
        self._z = torch.normal(torch.zeros(self.weight_dim), torch.ones(self.weight_dim))[:, None]
        self._sqrt_diag_psi_dot_z = torch.sqrt(self.diag_psi) * self._z
        return (self.F.mm(self._h) + self.c + self._sqrt_diag_psi_dot_z).squeeze(dim=1)

    def _accumulate_gradients(self, grad_weights: Tensor):
        """
        Accumulate gradients wrt the parameters of the variational distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).
        """
        self.c.grad += self._compute_gradient_wrt_c(grad_weights)
        self.F.grad += self._compute_gradient_wrt_F(grad_weights)
        self._log_diag_psi.grad += self._compute_gradient_wrt_log_diag_psi(grad_weights)

    def _compute_gradient_wrt_c(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the variational objective wrt the bias term of the factor analysis variational
        distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the variational objective wrt the bias term of the factor analysis variational
            distribution. Of shape (self.weight_dim, 1).
        """
        return -self._prior_grad_wrt_c + grad_weights

    def _compute_gradient_wrt_F(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the variational objective wrt the factors matrix of the factor analysis variational
        distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the variational objective wrt the factors matrix of the factor analysis variational
            distribution. Of shape (self.weight_dim, self.latent_dim).
        """
        loss_grad = self._compute_loss_gradient_wrt_F(grad_weights)

        return self._var_grad_wrt_F - self._prior_grad_wrt_F + loss_grad

    def _compute_loss_gradient_wrt_F(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the network's loss wrt the factors matrix.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the network's loss wrt the factors matrix. Of shape (self.weight_dim, self.latent_dim).
        """
        return grad_weights.mm(self._h.t())

    def _compute_gradient_wrt_log_diag_psi(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the variational objective wrt the logarithm of the diagonal of the noise covariance
        matrix of the factor analysis variational distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the variational objective wrt the logarithm of the diagonal of the noise covariance
            matrix of the factor analysis variational distribution. Of shape (self.weight_dim, 1).
        """
        loss_grad = self._compute_loss_gradient_wrt_log_diag_psi(grad_weights)
        return self._var_grad_wrt_log_diag_psi - self._prior_grad_wrt_log_diag_psi + loss_grad

    def _compute_loss_gradient_wrt_log_diag_psi(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the network's loss wrt the logarithm of the diagonal of the noise covariance matrix.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the network's loss wrt the logarithm of the diagonal of the noise covariance matrix. Of
            shape (self.weight_dim, 1).
        """
        return (1 / 2) * grad_weights * self._sqrt_diag_psi_dot_z

    def _update_variational_params(self):
        """
        Update the parameters of the factor analysis variational distribution.

        This is done by using the accumulated gradients to approximate the expected gradients and then performing a
        gradient step.

        After performing the updates, the gradients are reset to zero.
        """
        self._average_and_normalise_gradient(self.c)
        self._average_and_normalise_gradient(self.F)
        self._average_and_normalise_gradient(self._log_diag_psi)

        self._optimiser.step()
        self._optimiser.zero_grad()

        self.diag_psi = torch.exp(self._log_diag_psi)

    def _average_and_normalise_gradient(self, var: Variable):
        """
        Average the gradients accumulated in the variable by dividing by self.n_gradients_per_update and normalise if
        required.

        Args:
            var: The variable whose gradient to average and normalise.
        """
        var.grad /= self.n_gradients_per_update

        if self.max_grad_norm is not None:
            var.grad = normalise_gradient(var.grad, self.max_grad_norm)

    def _update_expected_gradients(self):
        """
        Update the expected gradients used in the algorithm which do not depend on the sampled network weights.
        """
        self._update_inverse_psi()
        self._update_A()
        self._update_B()
        self._update_sigma()
        self._update_C()
        self._update_s()
        self._update_variational_gradient_wrt_F()
        self._update_variational_gradient_wrt_log_diag_psi()
        self._update_prior_gradient_wrt_c()
        self._update_prior_gradient_wrt_F()
        self._update_prior_gradient_wrt_log_diag_psi()

    def _update_inverse_psi(self):
        """
        Invert the diagonal Gaussian noise covariance matrix.
        """
        self._diag_inv_psi = 1 / self.diag_psi

    def _update_A(self):
        """
        Update A = psi^(-1) * F.
        """
        self._A = self._diag_inv_psi * self.F

    def _update_B(self):
        """
        Update B = A^T * F.
        """
        self._B = self._A.t().mm(self.F)

    def _update_sigma(self):
        """
        Update sigma = (I + B)^(-1).
        """
        self._sigma = torch.linalg.inv(self._I + self._B)

    def _update_C(self):
        """
        Update C = sigma * (2B + I - (B + I) * B * sigma) * F^T.
        """
        B_times_sigma = self._B.mm(self._sigma)
        B_plus_I = self._B + self._I
        inner_term = 2 * self._B + self._I - B_plus_I.mm(B_times_sigma)
        self._C = (self._sigma.mm(inner_term)).mm(self.F.t())

    def _update_s(self):
        """
        Update s = diag(F * F^T + sigma - F * C).
        """
        inner_term = self.F * (self.F - self._C.t())
        sum_term = torch.sum(inner_term, dim=1, keepdim=True)
        self._s = self.diag_psi + sum_term

    def _update_variational_gradient_wrt_F(self):
        """
        Update d(variational distribution) / d(F) = A * (I - sigma * B - sigma) * B * sigma
        """
        inner_term = self._I - self._sigma.mm(self._B) - self._sigma
        self._var_grad_wrt_F = self._A.mm(inner_term).mm(self._B).mm(self._sigma)

    def _update_variational_gradient_wrt_log_diag_psi(self):
        """
        Update d(variational distribution) / d(log diag psi) = (1 / 2) * (psi ^(-1) * s - 1)
        """
        self._var_grad_wrt_log_diag_psi = (1 / 2) * (self._diag_inv_psi * self._s - 1)

    def _update_prior_gradient_wrt_c(self):
        """
        Update d(prior distribution) / d(c) = -precision * c
        """
        self._prior_grad_wrt_c = -self.precision * self.c

    def _update_prior_gradient_wrt_F(self):
        """
        Update d(prior distribution) / d(F) = -precision * F
        """
        self._prior_grad_wrt_F = -self.precision * self.F

    def _update_prior_gradient_wrt_log_diag_psi(self):
        """
        Update d(prior distribution) / d(log diag psi) = (-precision / 2) * diag_psi
        """
        self._prior_grad_wrt_log_diag_psi = (-self.precision / 2) * self.diag_psi

    def get_variational_mean(self) -> Tensor:
        """
        Get the mean of the factor analysis variational distribution.

        Returns:
            The mean vector. Of shape (self.weight_dim,).
        """
        return self.c.squeeze()

    def get_variational_covariance(self) -> Tensor:
        """
        Get the full covariance matrix of the factor analysis variational distribution.

        Note: if the network dimension is large, this may result in a memory error.

        Returns:
            The covariance matrix. Of shape (self.weight_dim, self.weight_dim).
        """
        psi = torch.diag(self.diag_psi.squeeze())
        return self.F.mm(self.F.t()) + psi
