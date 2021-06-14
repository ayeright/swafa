from abc import ABC, abstractmethod

import torch
from torch import Tensor


class OnlineFA(ABC):

    def __init__(self, observation_dim: int, latent_dim: int):
        self.d = observation_dim
        self.k = latent_dim
        self.F = torch.randn(self.d, self.k)
        self.diag_psi = torch.randn(self.d, 1)
        self.theta_hat = torch.zeros(self.d, 1)
        self.t = 0

    def calc_starter_values(self, theta: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        diag_inv_psi = self.invert_psi()
        d = self.centre_theta(theta)
        m, sigma = self.calc_variational_params(d, diag_inv_psi)
        return diag_inv_psi, d, m, sigma

    def centre_theta(self, theta: Tensor) -> Tensor:
        self.theta_hat = self.update_running_average(self.theta_hat, theta)
        return theta - self.theta_hat

    def invert_psi(self) -> Tensor:
        return 1 / self.diag_psi

    def calc_variational_params(self, d: Tensor, diag_inv_psi: Tensor) -> (Tensor, Tensor):
        C = (self.F * diag_inv_psi).t()
        sigma = self.calc_variational_sigma(C)
        m = self.calc_variational_mean(sigma, C, d)
        return m, sigma

    def calc_variational_sigma(self, C: Tensor) -> Tensor:
        return torch.linalg.inv(torch.eye(self.k) + C.mm(self.F))

    @staticmethod
    def calc_variational_mean(sigma: Tensor, C: Tensor, d: Tensor) -> Tensor:
        return sigma.mm(C.mm(d))

    def update_running_average(self, old_average: Tensor, new_obs: Tensor) -> Tensor:
        return old_average + (new_obs - old_average) / self.t

    @abstractmethod
    def update(self, theta: Tensor):
        ...


class OnlineGradientFA(OnlineFA):

    def __init__(self, observation_dim: int, latent_dim: int, learning_rate: float):
        super().__init__(observation_dim, latent_dim)
        self.alpha = learning_rate

    def update(self, theta: Tensor):
        diag_inv_psi, d, m, sigma = self.calc_starter_values(theta)
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


class OnlineEMFA(OnlineFA):

    def __init__(self, observation_dim: int, latent_dim: int):
        super().__init__(observation_dim, latent_dim)
        self.A_hat = torch.zeros(self.d, self.k)
        self.B_hat = torch.zeros(self.k, self.k)
        self.d_squared_hat = torch.zeros(self.d, 1)

    def update(self, theta: Tensor):
        diag_inv_psi, d, m, sigma = self.calc_starter_values(theta)
        self.update_A_hat(d, m)
        H = self.calc_H(sigma, m)
        self.update_F(H)
        self.update_psi(d, H)

    def update_A_hat(self, d: Tensor, m: Tensor):
        self.A_hat = self.update_running_average(self.A_hat, d.mm(m.t()))

    def calc_H(self, sigma: Tensor, m: Tensor) -> Tensor:
        self.update_B_hat(m)
        return sigma + self.B_hat

    def update_B_hat(self, m: Tensor):
        self.A_hat = self.update_running_average(self.A_hat, m.mm(m.t()))

    def update_F(self, H: Tensor):
        return self.A_hat.mm(torch.linalg.inv(H))

    def update_psi(self, d: Tensor, H: Tensor):
        self.update_d_squared_hat(d)
        self.diag_psi = self.d_squared_hat + torch.sum(self.F.mm(H) * self.F - 2 * self.F * self.A_hat, dim=1)

    def update_d_squared_hat(self, d: Tensor):
        self.A_hat = self.update_running_average(self.A_hat, d ** 2)
