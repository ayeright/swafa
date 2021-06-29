import numpy as np
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import SGD, Adam
from sklearn.decomposition import FactorAnalysis

from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis


def run_all_experiments():
    run_experiment(5, 3, 10000)


def run_experiment(observation_dim: int, latent_dim: int, n_samples: int):
    c = torch.randn(observation_dim)
    F = torch.randn(observation_dim, latent_dim)
    psi = torch.diag(torch.rand(observation_dim))
    covar = compute_covariance(F, psi)
    observations = sample_observations(c, F, psi, n_samples)

    mean_sklearn, covar_sklearn = solve_with_sklearn(observations, latent_dim)
    mean_online_gradients, covar_online_gradients = solve_with_online_gradients(observations, latent_dim)
    mean_online_em, covar_online_em = solve_with_online_em(observations, latent_dim)

    ll = compute_gaussian_log_likelihood(c, covar, observations)
    ll_sklearn = compute_gaussian_log_likelihood(mean_sklearn, covar_sklearn, observations)
    ll_online_gradients = compute_gaussian_log_likelihood(mean_online_gradients, covar_online_gradients, observations)
    ll_online_em = compute_gaussian_log_likelihood(mean_online_em, covar_online_em, observations)

    covar_distance_sklearn = torch.linalg.norm(covar - covar_sklearn)
    covar_distance_online_gradients = torch.linalg.norm(covar - covar_online_gradients)
    covar_distance_online_em = torch.linalg.norm(covar - covar_online_em)


def sample_observations(c: Tensor, F: Tensor, psi: Tensor, n_samples: int) -> Tensor:
    observation_dim, latent_dim = F.shape
    p_h = MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))
    p_psi = MultivariateNormal(loc=torch.zeros(observation_dim), covariance_matrix=psi)
    h = p_h.sample((n_samples,))  # (n_samples, latent_dim)
    noise = p_psi.sample((n_samples,))  # (n_samples, observation_dim)
    return h.mm(F.t()) + c.reshape(1, -1) + noise


def solve_with_sklearn(observations: Tensor, latent_dim: int) -> (Tensor, Tensor):
    fa = FactorAnalysis(n_components=latent_dim, svd_method='lapack')
    fa.fit(observations.numpy())
    mean = torch.from_numpy(fa.mean_)
    covar = torch.from_numpy(fa.get_covariance())
    return mean, covar


def solve_with_online_gradients(observations: Tensor, latent_dim: int) -> (Tensor, Tensor):
    observation_dim = observations.shape[1]
    fa = OnlineGradientFactorAnalysis(observation_dim, latent_dim, optimiser=Adam, optimiser_kwargs=dict(lr=1e-2))
    for theta in observations:
        fa.update(theta)
    mean = fa.c.squeeze()
    covar = compute_covariance(fa.F, fa.diag_psi)
    return mean, covar


def solve_with_online_em(observations: Tensor, latent_dim: int) -> (Tensor, Tensor):
    observation_dim = observations.shape[1]
    fa = OnlineEMFactorAnalysis(observation_dim, latent_dim)
    for theta in observations:
        fa.update(theta)
    mean = fa.c.squeeze()
    covar = compute_covariance(fa.F, fa.diag_psi)
    return mean, covar


def compute_covariance(F: Tensor, psi: Tensor) -> Tensor:
    FFt = F.mm(F.t())
    if len(psi.shape) == 1:
        return FFt + torch.diag(psi)
    return FFt + psi


def compute_gaussian_log_likelihood(mean: Tensor, covar: Tensor, X: Tensor) -> Tensor:
    n, d = X.shape
    inv_covar = torch.linalg.inv(covar)
    centred_X = X - mean.reshape(1, -1)
    unnormalised_log_likelihood = -0.5 * torch.sum(centred_X.mm(inv_covar) * centred_X, dim=1).mean()
    log_normalising_factor = -0.5 * (torch.logdet(covar) + d * np.log(2 * np.pi))
    return unnormalised_log_likelihood + log_normalising_factor


def main():
    run_all_experiments()


if __name__ == '__main__':
    main()
