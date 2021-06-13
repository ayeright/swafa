import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_


class OnlineFA(object):

    def __init__(self, observation_dim: int, latent_dim: int):
        self.d = observation_dim
        self.k = latent_dim
        self.F = torch.randn(self.d, self.k)
        self.diag_psi = torch.randn(self.d, 1)
        self.theta_hat = torch.zeros(self.d, 1)
        self.t = 0

    def centre_theta(self, theta: Tensor):
        self.theta_hat = self.update_running_average(self.theta_hat, theta)
        return theta - self.theta_hat

    def invert_psi(self):
        return 1 / self.diag_psi

    def calc_variational_params(self, d: Tensor, diag_inv_psi: Tensor):
        C = (self.F * diag_inv_psi).t()  # (k, d)
        sigma = self.calc_variational_sigma(C)  # (k, k)
        m = self.calc_variational_mean(sigma, C, d)  # (k, 1)
        return m, sigma

    def calc_variational_sigma(self, C: Tensor):
        return torch.linalg.inv(torch.eye(self.k) + C.mm(self.F))

    @staticmethod
    def calc_variational_mean(sigma: Tensor, C: Tensor, d: Tensor):
        return sigma.mm(C.mm(d))

    def update_running_average(self, old_average: Tensor, new_obs: Tensor):
        return old_average + (new_obs - old_average) / self.t


class OnlineGradientFA(OnlineFA):

    def __init__(self, observation_dim: int, latent_dim: int, learning_rate: float):
        super().__init__(observation_dim, latent_dim)
        self.alpha = learning_rate

    def update(self, theta: Tensor):
        d = self.centre_theta(theta)
        diag_inv_psi = self.invert_psi()
        m, sigma = self.calc_variational_params(d, diag_inv_psi)
        F_gradient = self.calc_F_gradient(diag_inv_psi, d, m, sigma)
        self.F = self.gradient_step(self.F, F_gradient)
        diag_psi_gradient = self.calc_psi_gradient(diag_inv_psi, d, m, sigma)
        self.diag_psi = self.gradient_step(self.diag_psi, diag_psi_gradient)

    def calc_F_gradient(self, diag_inv_psi: Tensor, d: Tensor, m: Tensor, sigma: Tensor):
        return Tensor([])

    def calc_psi_gradient(self, diag_inv_psi: Tensor, d: Tensor, m: Tensor, sigma: Tensor):
        return Tensor([])

    def gradient_step(self, x: Tensor, gradient: Tensor):
        return x + self.alpha * gradient



class OnlineEMFA(OnlineFA):

    def __init__(self, observation_dim: int, latent_dim: int):
        super().__init__(observation_dim, latent_dim)

    def update(self, theta: Tensor):
