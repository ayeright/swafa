from abc import ABC, abstractmethod

import torch
from torch import Tensor


class OnlineFactorAnalysis(ABC):
    """
    An abstract class used as a base for learning factor analysis (FA) models [1] online.

    Any concrete class which inherits from this class must implement the `update` method.

    The variable names used in this class generally matches those used in [1].

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

    def __init__(self, observation_dim: int, latent_dim: int):
        """
        Initialise the mean of the observed variables, the factor loading matrix and the Gaussian noise covariance
        matrix.

        Args:
            observation_dim: The size of the observed variable space.
            latent_dim: The size of the latent variable space.
        """
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.c = torch.zeros(observation_dim, 1)
        self.F = torch.randn(observation_dim, latent_dim)
        self.diag_psi = torch.randn(observation_dim, 1)
        self.t = 0

    def update_commons(self, theta: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
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
        m, sigma = self._calc_variational_params(d, diag_inv_psi)
        return d, diag_inv_psi, m, sigma

    def _update_c(self, theta: Tensor):
        """
        Update the running average of the observed variables.

        Args:
            theta: A single observation. Of shape (observation_dim, 1).
        """
        self.c = self.update_running_average(self.c, theta)

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

    def _calc_variational_params(self, d: Tensor, diag_inv_psi: Tensor) -> (Tensor, Tensor):
        """
        Calculate the mean and covariance of the optimal variational distribution given the centred observation.

        The optimal variational distribution is `p(h | theta, F, Psi)p(h)`. That is, the posterior distribution of the
        latent variables given the observation and the current values of `F` and `Psi`.

        This distribution is Gaussian with mean `m` and covariance `sigma`, as given in [1].

        Args:
            d: The centred observation. Of shape (observation_dim, 1).
            diag_inv_psi: The diagonal entries of the inverse of the noise covariance matrix. Of shape
                (observation_dim, 1).

        Returns:
            m: The mean of the optimal variational distribution. Of shape (latent_dim, 1).
            sigma: The covariance of the optimal variational distribution. Of shape (latent_dim, latent_dim).
        """
        C = (self.F * diag_inv_psi).t()
        sigma = self._calc_variational_covariance(C)
        m = self._calc_variational_mean(sigma, C, d)
        return m, sigma

    def _calc_variational_covariance(self, C: Tensor) -> Tensor:
        """
        Calculate the covariance of the optimal variational distribution.

        Args:
            C: The transpose of `F` right-multiplied by the inverse of `Psi`. Of shape (latent_dim, observation_dim).

        Returns:
           The covariance of the optimal variational distribution. Of shape (latent_dim, latent_dim).
        """
        I = torch.eye(self.latent_dim)
        return torch.linalg.inv(I + C.mm(self.F))

    @staticmethod
    def _calc_variational_mean(sigma: Tensor, C: Tensor, d: Tensor) -> Tensor:
        """
        Calculate the mean of the optimal variational distribution.

        Args:
            sigma: The covariance of the optimal variational distribution. Of shape (latent_dim, latent_dim).
            C: The transpose of `F` right-multiplied by the inverse of `Psi`. Of shape (latent_dim, observation_dim).
            d: The centred observation. Of shape (observation_dim, 1).

        Returns:
           The mean of the optimal variational distribution. Of shape (latent_dim, 1).
        """
        return sigma.mm(C.mm(d))

    def update_running_average(self, old_average: Tensor, new_observation: Tensor) -> Tensor:
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

    def __init__(self, observation_dim: int, latent_dim: int, learning_rate: float):
        super().__init__(observation_dim, latent_dim)
        self.alpha = learning_rate

    def update(self, theta: Tensor):
        d, diag_inv_psi, m, sigma = self.update_commons(theta)
        F_times_sigma_plus_m_mt = self.calc_F_times_sigma_plus_m_mt(m, sigma)
        self.update_F(diag_inv_psi, d, m, F_times_sigma_plus_m_mt)
        self.update_psi(diag_inv_psi, d, m, F_times_sigma_plus_m_mt)

    def update_F(self, diag_inv_psi: Tensor, d: Tensor, m: Tensor, sigma: Tensor):
        gradient_wrt_F = self.calc_gradient_wrt_F(diag_inv_psi, d, m, sigma)
        self.F = self.gradient_step(self.F, gradient_wrt_F)

    def update_psi(self, diag_inv_psi: Tensor, d: Tensor, m: Tensor, sigma: Tensor):
        diag_gradient_wrt_psi = self.calc_gradient_wrt_psi(diag_inv_psi, d, m, sigma)
        self.diag_psi = self.gradient_step(self.diag_psi, diag_gradient_wrt_psi)

    def calc_gradient_wrt_F(self, diag_inv_psi: Tensor, d: Tensor, m: Tensor, F_times_sigma_plus_m_mt: Tensor,
                            ) -> Tensor:
        return diag_inv_psi * (d.mm(m.t()) - F_times_sigma_plus_m_mt)

    def calc_gradient_wrt_psi(self, diag_inv_psi: Tensor, d: Tensor, m: Tensor, F_times_sigma_plus_m_mt: Tensor,
                              ) -> Tensor:
        E = d ** 2 - 2 * d * self.F.mm(m) + torch.sum(F_times_sigma_plus_m_mt * self.F, dim=1)
        return ((diag_inv_psi ** 2) * E - diag_inv_psi) / 2

    def calc_F_times_sigma_plus_m_mt(self, m: Tensor, sigma: Tensor) -> Tensor:
        return self.F.mm(sigma + m.mm(m.t()))

    def gradient_step(self, x: Tensor, gradient: Tensor):
        return x + self.alpha * gradient


class OnlineEMFactorAnalysis(OnlineFactorAnalysis):

    def __init__(self, observation_dim: int, latent_dim: int):
        super().__init__(observation_dim, latent_dim)
        self.A_hat = torch.zeros(observation_dim, latent_dim)
        self.B_hat = torch.zeros(latent_dim, latent_dim)
        self.d_squared_hat = torch.zeros(observation_dim, 1)

    def update(self, theta: Tensor):
        d, diag_inv_psi, m, sigma = self.update_commons(theta)
        H = self.calc_H(sigma, m)
        self.update_A_hat(d, m)
        self.update_F(H)
        self.update_psi(d, H)

    def calc_H(self, sigma: Tensor, m: Tensor) -> Tensor:
        self.update_B_hat(m)
        return sigma + self.B_hat

    def update_B_hat(self, m: Tensor):
        self.B_hat = self.update_running_average(self.B_hat, m.mm(m.t()))

    def update_A_hat(self, d: Tensor, m: Tensor):
        self.A_hat = self.update_running_average(self.A_hat, d.mm(m.t()))

    def update_F(self, H: Tensor):
        return self.A_hat.mm(torch.linalg.inv(H))

    def update_psi(self, d: Tensor, H: Tensor):
        self.update_d_squared_hat(d)
        self.diag_psi = self.d_squared_hat + torch.sum(self.F.mm(H) * self.F - 2 * self.F * self.A_hat, dim=1)

    def update_d_squared_hat(self, d: Tensor):
        self.A_hat = self.update_running_average(self.A_hat, d ** 2)
