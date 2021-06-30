from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import Adam
from sklearn.decomposition import FactorAnalysis

from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis


def run_all_experiments(experiments_config: List[dict]) -> pd.DataFrame:
    results = []
    for config in experiments_config:
        results.append(
            run_experiment(
                config['observation_dim'],
                config['latent_dim'],
                config['spectrum_range'],
                config['n_samples'],
            )
        )
    return pd.DataFrame(results)


def run_experiment(observation_dim: int, latent_dim: int, spectrum_range: List[int], n_samples: int
                   ) -> Dict[str, float]:
    results = dict()

    c, F, psi = generate_fa_model(observation_dim, latent_dim, spectrum_range)
    covar = compute_covariance(F, psi)
    observations = sample_observations(c, F, psi, n_samples)

    mean_sklearn, covar_sklearn = solve_with_sklearn(observations, latent_dim)
    mean_online_gradients, covar_online_gradients = solve_with_online_gradients(observations, latent_dim)
    mean_online_em, covar_online_em = solve_with_online_em(observations, latent_dim)

    results['covar_norm'] = torch.linalg.norm(covar)
    results['covar_distance_sklearn'] = torch.linalg.norm(covar - covar_sklearn)
    results['covar_distance_online_gradients'] = torch.linalg.norm(covar - covar_online_gradients)
    results['covar_distance_online_em'] = torch.linalg.norm(covar - covar_online_em)

    results['ll'] = compute_gaussian_log_likelihood(c, covar, observations)
    results['ll_sklearn'] = compute_gaussian_log_likelihood(mean_sklearn, covar_sklearn, observations)
    results['ll_online_gradients'] = compute_gaussian_log_likelihood(mean_online_gradients, covar_online_gradients,
                                                                     observations)
    results['ll_online_em'] = compute_gaussian_log_likelihood(mean_online_em, covar_online_em, observations)

    return results


def generate_fa_model(observation_dim: int, latent_dim: int, spectrum_range: List[int]) -> (Tensor, Tensor, Tensor):
    c = torch.randn(observation_dim)
    F, spectrum = generate_factors(observation_dim, latent_dim, spectrum_range)
    psi = generate_noise_covariance(observation_dim, spectrum)
    return c, F, psi


def generate_factors(observation_dim: int, latent_dim: int, spectrum_range: List[int]) -> (Tensor, Tensor):
    A = torch.randn(observation_dim, observation_dim)
    M = A.mm(A.t())
    _, V = torch.linalg.eigh(M)
    Vk = V[:, :latent_dim]
    spectrum = torch.FloatTensor(observation_dim, 1).uniform_(*spectrum_range)
    F = Vk * torch.sqrt(spectrum)
    return F, spectrum


def generate_noise_covariance(observation_dim: int, spectrum: Tensor) -> Tensor:
    diag_psi = torch.FloatTensor(observation_dim).uniform_(0, spectrum.max())
    return torch.diag(diag_psi)


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
    results = run_all_experiments()


if __name__ == '__main__':
    main()
