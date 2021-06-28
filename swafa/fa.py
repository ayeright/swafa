from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer, Adam
from torch.autograd import Variable


class OnlineFactorAnalysis(ABC):
    """
    An abstract class used as a base for learning factor analysis (FA) models [1] online.

    Any concrete class which inherits from this class must implement the `update` method.

    The variable names used in this class generally match those used in [1].

    Factor loading matrix `F` is initialised to be a matrix with 1s on the diagonal and zero mean Gaussian noise
    everywhere else.

    Diagonal covariance matrix `Psi` is initialised to be the identity matrix.

    Args:
        observation_dim: The size of the observed variable space.
        latent_dim: The size of the latent variable space.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix.
        device: The device (CPU or GPU) on which to perform the computation. If `None`, uses the device for the default
            tensor type.
        random_seed: The random seed for reproducibility.

    Attributes:
        observation_dim: The size of the observed variable space. An integer.
        latent_dim: The size of the latent variable space. An integer.
        t: The current time step, or equivalently, the number of observations seen. An integer which starts off as 0.
        c: The mean of the observed variables. A Tensor of shape (observation_dim, 1).
        F: The factor loading matrix. A Tensor of shape (observation_dim, latent_dim).
        diag_psi: The diagonal entries of the Gaussian noise covariance matrix, usually referred to as `Psi`. A Tensor
            of shape (observation_dim, 1).

    References:
        [1] David Barber. Bayesian Reasoning and Machine Learning. Cambridge University Press, 2012.
    """

    def __init__(self, observation_dim: int, latent_dim: int, init_factors_noise_std: float = 1e-3,
                 device: Optional[torch.device] = None, random_seed: int = 0):
        torch.manual_seed(random_seed)
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.t = 0
        self.c = torch.zeros(observation_dim, 1, device=device)
        self.F = self._init_F(init_factors_noise_std, device)
        self.diag_psi = torch.ones(observation_dim, 1, device=device)
        self._diag_inv_psi = None
        self._d = None
        self._m = None
        self._sigma = None
        self._I = torch.eye(latent_dim, device=device)

    def _init_F(self, noise_std: float, device: Optional[torch.device] = None) -> Tensor:
        """
        Initialise the factor loading matrix.

        Initialised to be a matrix with 1s on the diagonal and zero mean Gaussian noise everywhere else.

        Args:
            noise_std: The standard deviation of the noise in the off-diagonal entries.
            device: The device (CPU or GPU) on which to perform the computation. If `None`, uses the device for the
                default tensor type.

        Returns:
            The initial factor loading matrix. Of shape (observation_dim, latent_dim).
        """
        I = torch.eye(self.observation_dim, self.latent_dim, device=device)
        off_diagonal_noise = torch.normal(0, noise_std, (self.observation_dim, self.latent_dim)).to(device)
        return I + (1 - I) * off_diagonal_noise

    def _update_commons(self, theta: Tensor):
        """
        Given an observation, perform updates which are common to all online FA algorithms.

        Specifically, update self.t, self.c, self._d, self._diag_inv_psi, self._m and self._sigma given the new
        observation and the current values of self.F and self.diag_psi.

        Args:
            theta: A single observation of shape (observation_dim,) or (observation_dim, 1).
        """
        theta = theta.reshape(-1, 1)
        self.t += 1
        self._update_observation_mean(theta)
        self._update_centred_observation(theta)
        self._update_inverse_psi()
        self._update_latent_posterior_params()

    def _update_observation_mean(self, theta: Tensor):
        """
        Update the running average of the observed variables.

        Args:
            theta: A single observation. Of shape (observation_dim, 1).
        """
        self.c = self._update_running_average(self.c, theta)

    def _update_centred_observation(self, theta: Tensor):
        """
        Centre the observation by subtracting the mean of all observations.

        Args:
            theta: A single observation. Of shape (observation_dim, 1).
        """
        self._d = theta - self.c

    def _update_inverse_psi(self):
        """
        Invert the diagonal Gaussian noise covariance matrix.
        """
        self._diag_inv_psi = 1 / self.diag_psi

    def _update_latent_posterior_params(self):
        """
        Update the mean and covariance of the posterior distribution of the latent variables.

        The distribution is `p(h | theta, F, Psi)p(h) = N(m, sigma)`, given the current observation and the current
        values of `F` and `Psi`.
        """
        C = (self.F * self._diag_inv_psi).t()
        self._update_latent_posterior_covariance(C)
        self._update_latent_posterior_mean(C)

    def _update_latent_posterior_covariance(self, C: Tensor):
        """
        Update the covariance of the posterior distribution of the latent variables.

        Args:
            C: The transpose of `F` right-multiplied by the inverse of `Psi`. Of shape (latent_dim, observation_dim).
        """
        self._sigma = torch.linalg.inv(self._I + C.mm(self.F))

    def _update_latent_posterior_mean(self, C: Tensor):
        """
        Update the mean of the posterior distribution of the latent variables.

        Args:
            C: The transpose of `F` right-multiplied by the inverse of `Psi`. Of shape (latent_dim, observation_dim).
        """
        self._m = self._sigma.mm(C.mm(self._d))

    def _update_running_average(self, old_average: Tensor, new_observation: Tensor) -> Tensor:
        """
        Update the running average given a new observation.

        Args:
            old_average: The average up until the current time step.
            new_observation: The observation to use to update the average.

        Returns:
            The updated running average.
        """
        return old_average + (new_observation - old_average) / self.t

    @abstractmethod
    def update(self, theta: Tensor):
        """
        Given a new observation, update the parameters of the FA model.

        Args:
            theta: A single observation of shape (observation_dim,) or (observation_dim, 1).
        """
        ...


class OnlineGradientFactorAnalysis(OnlineFactorAnalysis):
    """
    Implementation of online stochastic gradient factor analysis (FA) from [1].

    The variable names used in this class generally match those used in [1].

    Args:
        observation_dim: The size of the observed variable space.
        latent_dim: The size of the latent variable space.
        optimiser: The class of the optimiser to use for gradient updates.
        optimiser_kwargs: Keyword arguments for the optimiser. If not given, will default to dict(lr=1e-3).
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix.
        device: The device (CPU or GPU) on which to perform the computation. If `None`, uses the device for the default
            tensor type.
        random_seed: The random seed for reproducibility.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """

    def __init__(self, observation_dim: int, latent_dim: int, optimiser: Optimizer = Adam,
                 optimiser_kwargs: Optional[dict] = None, init_factors_noise_std: float = 1e-3,
                 device: Optional[torch.device] = None, random_seed: int = 0):
        super().__init__(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std, device=device,
                         random_seed=random_seed)
        optimiser_kwargs = optimiser_kwargs or dict(lr=1e-3)
        self.F = Variable(self.F, requires_grad=False)  # we will compute our own gradients
        self._log_diag_psi = Variable(torch.log(self.diag_psi), requires_grad=False)
        self._F_times_sigma_plus_m_mt = None
        self._gradient_wrt_F = None
        self._gradient_wrt_diag_psi = None
        self._gradient_wrt_log_diag_psi = None
        self._optimiser = optimiser([self.F, self._log_diag_psi], **optimiser_kwargs)

    def update(self, theta: Tensor):
        """
        Given a new observation, update the parameters of the FA model.

        Args:
            theta: A single observation of shape (observation_dim,) or (observation_dim, 1).
        """
        self._update_commons(theta)
        self._update_F_times_sigma_plus_m_mt()
        self._update_gradient_wrt_F()
        self._update_gradient_wrt_log_psi()
        self._gradient_step()
        self.diag_psi = torch.exp(self._log_diag_psi)

    def _update_F_times_sigma_plus_m_mt(self):
        """
        Update the value of `F(sigma + mm^T)`.

        This quantity is used multiple times in the gradient calculations, so it is more efficient to compute it only
        once.
        """
        self._F_times_sigma_plus_m_mt = self.F.mm(self._sigma + self._m.mm(self._m.t()))

    def _update_gradient_wrt_F(self):
        """
        Update the value of the gradient of the log-likelihood wrt the factor loading matrix.
        """
        self._gradient_wrt_F = self._diag_inv_psi * (self._d.mm(self._m.t()) - self._F_times_sigma_plus_m_mt)

    def _update_gradient_wrt_log_psi(self):
        """
        Update the value of the gradient of the log-likelihood wrt the logarithm of the diagonal entries of the Gaussian
        noise covariance matrix.
        """
        self._update_gradient_wrt_psi()
        self._gradient_wrt_log_diag_psi = self._gradient_wrt_diag_psi * self.diag_psi

    def _update_gradient_wrt_psi(self):
        """
        Update the value of the gradient of the log-likelihood wrt the diagonal entries of the Gaussian noise covariance
        matrix.
        """
        E = self._d ** 2 \
            - 2 * self._d * self.F.mm(self._m) \
            + torch.sum(self._F_times_sigma_plus_m_mt * self.F, dim=1, keepdim=True)
        self._gradient_wrt_diag_psi = ((self._diag_inv_psi ** 2) * E - self._diag_inv_psi) / 2

    def _gradient_step(self):
        """
        Perform a gradient step to update self.F and self._log_diag_psi.

        Goal is to maximise the log-likelihood, but Torch optimisers are designed to minimise. So multiply the gradients
        by -1 before performing the updates.
        """
        self.F.grad = -self._gradient_wrt_F
        self._log_diag_psi.grad = -self._gradient_wrt_log_diag_psi
        self._optimiser.step()


class OnlineEMFactorAnalysis(OnlineFactorAnalysis):
    """
    Implementation of online expectation maximisation for factor analysis (FA) from [1].

    The variable names used in this class generally match those used in [1].

    Args:
        observation_dim: The size of the observed variable space.
        latent_dim: The size of the latent variable space.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix.
        device: The device (CPU or GPU) on which to perform the computation. If `None`, uses the device for the default
            tensor type.
        random_seed: The random seed for reproducibility.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """

    def __init__(self, observation_dim: int, latent_dim: int, init_factors_noise_std: float = 1e-3,
                 device: Optional[torch.device] = None, random_seed: int = 0):
        super().__init__(observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std, device=device,
                         random_seed=random_seed)
        self._A_hat = torch.zeros(observation_dim, latent_dim, device=device)
        self._B_hat = torch.zeros(latent_dim, latent_dim, device=device)
        self._H_hat = None
        self._d_squared_hat = torch.zeros(observation_dim, 1, device=device)

    def update(self, theta: Tensor):
        """
        Given a new observation, update the running averages and the parameters of the FA model.

        Args:
            theta: A single observation of shape (observation_dim,) or (observation_dim, 1).
        """
        self._update_commons(theta)
        self._update_H_hat()
        self._update_A_hat()
        if self.t > 1:  # to avoid computing the inverse of H_hat, which is zero on the first iteration
            self._update_F()
            self._update_psi()

    def _update_H_hat(self):
        """
        Update the sum of the latent posterior covariance matrix and the running average of `mm^t`.
        """
        self._update_B_hat()
        self._H_hat = self._sigma + self._B_hat

    def _update_B_hat(self):
        """
        Update the running average of `mm^t`.
        """
        self._B_hat = self._update_running_average(self._B_hat, self._m.mm(self._m.t()))

    def _update_A_hat(self):
        """
        Update the running average of `dm^t`.
        """
        self._A_hat = self._update_running_average(self._A_hat, self._d.mm(self._m.t()))

    def _update_F(self):
        """
        Update the factor loading matrix.
        """
        self.F = self._A_hat.mm(torch.linalg.inv(self._H_hat))

    def _update_psi(self):
        """
        Update the diagonal entries of the Gaussian noise covariance matrix.
        """
        self._update_d_squared_hat()
        self.diag_psi = self._d_squared_hat \
            + torch.sum(self.F.mm(self._H_hat) * self.F - 2 * self.F * self._A_hat, dim=1, keepdim=True)

    def _update_d_squared_hat(self):
        """
        Update the running average of `d^2`.
        """
        self._d_squared_hat = self._update_running_average(self._d_squared_hat, self._d ** 2)
