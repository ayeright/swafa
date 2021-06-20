from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class OnlineFactorAnalysis(ABC):
    """
    An abstract class used as a base for learning factor analysis (FA) models [1] online.

    Any concrete class which inherits from this class must implement the `update` method.

    The variable names used in this class generally match those used in [1].

    Args:
        observation_dim: The size of the observed variable space.
        latent_dim: The size of the latent variable space.
        device: The device (CPU or GPU) on which to perform the computation. If `None`, uses the device for the default
            tensor type.

    Attributes:
        observation_dim: The size of the observed variable space. An integer.
        latent_dim: The size of the latent variable space. An integer.
        c: The mean of the observed variables. A Tensor of shape (observation_dim, 1).
        F: The factor loading matrix. A Tensor of shape (observation_dim, latent_dim).
        diag_psi: The diagonal entries of the Gaussian noise covariance matrix, usually referred to as `Psi`. A Tensor
            of shape (observation_dim, 1).
        t: The current time step, or equivalently, the number of observations seen. An integer which starts off as 0.

    References:
        [1] David Barber. Bayesian Reasoning and Machine Learning. Cambridge University Press, 2012.
    """

    def __init__(self, observation_dim: int, latent_dim: int, device: Optional[torch.device] = None):
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.c = torch.zeros(observation_dim, 1, device=device)
        self.F = torch.randn(observation_dim, latent_dim, device=device)
        self.diag_psi = torch.ones(observation_dim, 1, device=device)
        self.t = 0

    def _update_commons(self, theta: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Given an observation, perform updates which are common to all online FA algorithms.

        That is, increment the current time step and update the running mean of the observed variables.

        Also, calculate and return variables which are likely to be needed by all online FA algorithms.

        Args:
            theta: A single observation of shape (observation_dim,) or (observation_dim, 1).

        Returns:
            d: The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
            diag_inv_psi: The diagonal entries of the inverse of the Gaussian noise covariance matrix. Of shape
                (observation_dim, 1).
            m: The mean of `p(h | theta, F, Psi)p(h)`. That is, the posterior distribution of the latent variables given
                the observation and the current values of `F` and `Psi`. Of shape (latent_dim, 1).
            sigma: The covariance of `p(h | theta, F, Psi)p(h)`. Of shape (latent_dim, latent_dim).
        """
        self.t += 1
        theta = theta.reshape(-1, 1)
        self._update_c(theta)
        d = self._centre_observation(theta)
        diag_inv_psi = self._invert_psi()
        m, sigma = self._calc_latent_posterior_params(d, diag_inv_psi)
        return d, diag_inv_psi, m, sigma

    def _update_c(self, theta: Tensor):
        """
        Update the running average of the observed variables.

        Args:
            theta: A single observation. Of shape (observation_dim, 1).
        """
        self.c = self._update_running_average(self.c, theta)

    def _centre_observation(self, theta: Tensor) -> Tensor:
        """
        Centre the observation by subtracting the mean of all observations.

        Args:
            theta: A single observation. Of shape (observation_dim, 1).

        Returns:
            The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
        """
        return theta - self.c

    def _invert_psi(self) -> Tensor:
        """
        Invert the diagonal Gaussian noise covariance matrix.

        Returns:
            The diagonal entries of the inverse of the noise covariance matrix. Of shape (observation_dim, 1).
        """
        return 1 / self.diag_psi

    def _calc_latent_posterior_params(self, d: Tensor, diag_inv_psi: Tensor) -> (Tensor, Tensor):
        """
        Calculate the mean and covariance of the posterior distribution of the latent variables.

        The distribution is `p(h | theta, F, Psi)p(h) = N(m, sigma)`, given the observation and the current values of
        `F` and `Psi`.

        Args:
            d: The centred observation. Of shape (observation_dim, 1).
            diag_inv_psi: The diagonal entries of the inverse of the noise covariance matrix. Of shape
                (observation_dim, 1).

        Returns:
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
            sigma: The covariance of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, latent_dim).
        """
        C = (self.F * diag_inv_psi).t()
        sigma = self._calc_latent_posterior_covariance(C)
        m = self._calc_latent_posterior_mean(sigma, C, d)
        return m, sigma

    def _calc_latent_posterior_covariance(self, C: Tensor) -> Tensor:
        """
        Calculate the covariance of the posterior distribution of the latent variables.

        Args:
            C: The transpose of `F` right-multiplied by the inverse of `Psi`. Of shape (latent_dim, observation_dim).

        Returns:
           The covariance of the posterior distribution of the latent variables given the observation. Of shape
            (latent_dim, latent_dim).
        """
        I = torch.eye(self.latent_dim)
        return torch.linalg.inv(I + C.mm(self.F))

    @staticmethod
    def _calc_latent_posterior_mean(sigma: Tensor, C: Tensor, d: Tensor) -> Tensor:
        """
        Calculate the mean of the posterior distribution of the latent variables.

        Args:
            sigma: The covariance of the optimal variational distribution. Of shape (latent_dim, latent_dim).
            C: The transpose of `F` right-multiplied by the inverse of `Psi`. Of shape (latent_dim, observation_dim).
            d: The centred observation. Of shape (observation_dim, 1).

        Returns:
           The mean of the posterior distribution of the latent variables given the observation. Of shape
           (latent_dim, 1).
        """
        return sigma.mm(C.mm(d))

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

    Attributes:
        observation_dim: The size of the observed variable space. An integer.
        latent_dim: The size of the latent variable space. An integer.
        c: The mean of the observed variables. A Tensor of shape (observation_dim, 1).
        F: The factor loading matrix. A Tensor of shape (observation_dim, latent_dim).
        diag_psi: The diagonal entries of the Gaussian noise covariance matrix, usually referred to as `Psi`. A Tensor
            of shape (observation_dim, 1).
        log_diag_psi: The logarithm of diag_psi. A Tensor of shape (observation_dim, 1).
        t: The current time step, or equivalently, the number of observations seen. An integer which starts off as 0.
        alpha: The learning rate for gradient updates. A float.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """

    def __init__(self, observation_dim: int, latent_dim: int, learning_rate: float = 1e-3,
                 device: Optional[torch.device] = None):
        """
        Initialise the parameters of the FA model.

        Args:
            observation_dim: The size of the observed variable space.
            latent_dim: The size of the latent variable space.
            learning_rate: The learning rate for gradient updates.
            device: The device (CPU or GPU) on which to perform the computation. If `None`, uses the device for the
                default tensor type.
        """
        super().__init__(observation_dim, latent_dim, device=device)
        self.log_diag_psi = torch.log(self.diag_psi)
        self.alpha = learning_rate

    def update(self, theta: Tensor):
        """
        Given a new observation, update the parameters of the FA model.

        Args:
            theta: A single observation of shape (observation_dim,) or (observation_dim, 1).
        """
        d, diag_inv_psi, m, sigma = self._update_commons(theta)
        F_times_sigma_plus_m_mt = self._calc_F_times_sigma_plus_m_mt(m, sigma)
        gradient_wrt_F = self._calc_gradient_wrt_F(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        gradient_wrt_log_diag_psi = self._calc_gradient_wrt_log_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        self.F = self._gradient_ascent_step(self.F, gradient_wrt_F)
        self.log_diag_psi = self._gradient_ascent_step(self.log_diag_psi, gradient_wrt_log_diag_psi)
        self.diag_psi = torch.exp(self.log_diag_psi)

    def _calc_F_times_sigma_plus_m_mt(self, m: Tensor, sigma: Tensor) -> Tensor:
        """
        Calculate `F(sigma + mm^T)`.

        This quantity is used multiple times in the gradient calculations, so it is more efficient to compute it only
        once.

        Args:
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
            sigma: The covariance of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, latent_dim).

        Returns:
            `F(sigma + mm^T)`. Of shape (observation_dim, latent_dim).
        """
        return self.F.mm(sigma + m.mm(m.t()))

    def _calc_gradient_wrt_F(self, d: Tensor, diag_inv_psi: Tensor, m: Tensor, F_times_sigma_plus_m_mt: Tensor,
                             ) -> Tensor:
        """
        Calculate the gradient of the log-likelihood wrt the factor loading matrix.

        Args:
            d: The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
            diag_inv_psi: The diagonal entries of the inverse of the Gaussian noise covariance matrix. Of shape
                (observation_dim, 1).
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
            F_times_sigma_plus_m_mt: `F(sigma + mm^T)`. Of shape (observation_dim, latent_dim).

        Returns:
            The gradient of the log-likelihood wrt the factor loading matrix. Of shape (observation_dim, latent_dim).
        """
        return diag_inv_psi * (d.mm(m.t()) - F_times_sigma_plus_m_mt)

    def _calc_gradient_wrt_log_psi(self, d: Tensor, diag_inv_psi: Tensor, m: Tensor, F_times_sigma_plus_m_mt: Tensor,
                                   ) -> Tensor:
        """
        Calculate the gradient of the log-likelihood wrt the logarithm of the diagonal entries of the Gaussian noise
        covariance matrix.

        Args:
            d: The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
            diag_inv_psi: The diagonal entries of the inverse of the Gaussian noise covariance matrix. Of shape
                (observation_dim, 1).
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
            F_times_sigma_plus_m_mt: `F(sigma + mm^T)`. Of shape (observation_dim, latent_dim).

        Returns:
            The gradient of the log-likelihood wrt the logarithm of the diagonal elements of the Gaussian noise
            covariance matrix. Of shape (observation_dim, 1).
        """
        gradient_wrt_diag_psi = self._calc_gradient_wrt_psi(d, diag_inv_psi, m, F_times_sigma_plus_m_mt)
        return gradient_wrt_diag_psi * self.diag_psi

    def _calc_gradient_wrt_psi(self, d: Tensor, diag_inv_psi: Tensor, m: Tensor, F_times_sigma_plus_m_mt: Tensor,
                               ) -> Tensor:
        """
        Calculate the gradient of the log-likelihood wrt the diagonal entries of the Gaussian noise covariance matrix.

        Args:
            d: The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
            diag_inv_psi: The diagonal entries of the inverse of the Gaussian noise covariance matrix. Of shape
                (observation_dim, 1).
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
            F_times_sigma_plus_m_mt: `F(sigma + mm^T)`. Of shape (observation_dim, latent_dim).

        Returns:
            The gradient of the log-likelihood wrt the diagonal elements of the Gaussian noise covariance matrix. Of
            shape (observation_dim, 1).
        """
        E = d ** 2 \
            - 2 * d * self.F.mm(m) \
            + torch.sum(F_times_sigma_plus_m_mt * self.F, dim=1, keepdim=True)
        return ((diag_inv_psi ** 2) * E - diag_inv_psi) / 2

    def _gradient_ascent_step(self, x: Tensor, gradient: Tensor):
        """
        Update the input by taking a step in the direction of the positive gradient.

        The size of the step is controlled by the learning rate.

        Args:
            x: The input to update using the gradient. Of arbitrary shape.
            gradient: The gradient wrt x. Shape must be the same as x.

        Returns:
            The updated input.
        """
        return x + self.alpha * gradient


class OnlineEMFactorAnalysis(OnlineFactorAnalysis):
    """
    Implementation of online expectation maximisation for factor analysis (FA) from [1].

    The variable names used in this class generally match those used in [1].

    Attributes:
        observation_dim: The size of the observed variable space. An integer.
        latent_dim: The size of the latent variable space. An integer.
        c: The mean of the observed variables. A Tensor of shape (observation_dim, 1).
        F: The factor loading matrix. A Tensor of shape (observation_dim, latent_dim).
        diag_psi: The diagonal entries of the Gaussian noise covariance matrix, usually referred to as `Psi`. A Tensor
            of shape (observation_dim, 1).
        t: The current time step, or equivalently, the number of observations seen. An integer which starts off as 0.
        A_hat: The running average of `dm^t`. A Tensor of shape (observation_dim, latent_dim).
        B_hat: The running average of `mm^t`. A Tensor of shape (latent_dim, latent_dim).
        d_squared_hat: The running average of `d^2`. A Tensor of shape (observation_dim, 1).

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """

    def __init__(self, observation_dim: int, latent_dim: int, device: Optional[torch.device] = None):
        """
        Initialise the parameters of the FA model and the running averages.

        Args:
            observation_dim: The size of the observed variable space.
            latent_dim: The size of the latent variable space.
            device: The device (CPU or GPU) on which to perform the computation. If `None`, uses the device for the
                default tensor type.
        """
        super().__init__(observation_dim, latent_dim, device=device)
        self.A_hat = torch.zeros(observation_dim, latent_dim, device=device)
        self.B_hat = torch.zeros(latent_dim, latent_dim, device=device)
        self.d_squared_hat = torch.zeros(observation_dim, 1, device=device)

    def update(self, theta: Tensor):
        """
        Given a new observation, update the running averages and the parameters of the FA model.

        Args:
            theta: A single observation of shape (observation_dim,) or (observation_dim, 1).
        """
        d, diag_inv_psi, m, sigma = self._update_commons(theta)
        H = self._calc_H(m, sigma)
        self._update_A_hat(d, m)
        self._update_F(H)
        self._update_psi(d, H)

    def _calc_H(self, m: Tensor, sigma: Tensor) -> Tensor:
        """
        Calculate the sum of the latent posterior covariance matrix and the running average of `mm^t`.

        Args:
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
            sigma: The covariance of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, latent_dim).

        Returns:
            The sum of the latent posterior covariance matrix and the running average of `mm^t`. Of shape
            (latent_dim, latent_dim).
        """
        self._update_B_hat(m)
        return sigma + self.B_hat

    def _update_B_hat(self, m: Tensor):
        """
        Update the running average of `mm^t`.

        Args:
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
        """
        self.B_hat = self._update_running_average(self.B_hat, m.mm(m.t()))

    def _update_A_hat(self, d: Tensor, m: Tensor):
        """
        Update the running average of `dm^t`.

        Args:
            d: The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
            m: The mean of the posterior distribution of the latent variables given the observation. Of shape
                (latent_dim, 1).
        """
        self.A_hat = self._update_running_average(self.A_hat, d.mm(m.t()))

    def _update_F(self, H: Tensor):
        """
        Update the factor loading matrix.

        Args:
            H: The sum of the latent posterior covariance matrix and the running average of `mm^t`. Of shape
                (latent_dim, latent_dim).
        """
        return self.A_hat.mm(torch.linalg.inv(H))

    def _update_psi(self, d: Tensor, H: Tensor):
        """
        Update the diagonal entries of the Gaussian noise covariance matrix.

        Args:
            d: The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
            H: The sum of the latent posterior covariance matrix and the running average of `mm^t`. Of shape
                (latent_dim, latent_dim).
        """
        self._update_d_squared_hat(d)
        self.diag_psi = self.d_squared_hat + \
            torch.sum(self.F.mm(H) * self.F - 2 * self.F * self.A_hat, dim=1, keepdim=True)

    def _update_d_squared_hat(self, d: Tensor):
        """
        Update the running average of `d^2`.

        Args:
            d: The centred observation. That is, the current observation minus the mean of all observations. Of shape
                (observation_dim, 1).
        """
        self.d_squared_hat = self._update_running_average(self.d_squared_hat, d ** 2)
