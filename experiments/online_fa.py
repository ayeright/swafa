import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import SGD
from sklearn.decomposition import FactorAnalysis
import yaml
import click

from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis, OnlineFactorAnalysis


def run_all_experiments(experiments_config: List[dict], n_trials: int, init_factors_noise_std: float,
                        gradient_optimiser_kwargs: dict) -> pd.DataFrame:
    results = []
    for i_experiment, config in enumerate(experiments_config):
        print(f'Running experiment {i_experiment + 1} of {len(experiments_config)}...')
        for i_trial in range(n_trials):
            print(f'Running trial {i_trial + 1} of {n_trials}...')

            trial_results = run_experiment(
                config['observation_dim'],
                config['latent_dim'],
                config['spectrum_range'],
                config['n_samples'],
                init_factors_noise_std,
                gradient_optimiser_kwargs,
                samples_random_seed=i_trial,
                models_random_seed=i_trial + 1,
            )
            trial_results['experiment'] = i_experiment + 1
            trial_results['trial'] = i_trial + 1
            results.append(trial_results)

    return pd.DataFrame(results)


def run_experiment(observation_dim: int, latent_dim: int, spectrum_range: List[int], n_samples: int,
                   init_factors_noise_std: float, gradient_optimiser_kwargs: dict, samples_random_seed: int,
                   models_random_seed: int) -> Dict[str, float]:
    results = dict(
        observation_dim=observation_dim,
        latent_dim=latent_dim,
        spectrum_min=spectrum_range[0],
        spectrum_max=spectrum_range[1],
        n_samples=n_samples,
    )

    mean_true, covar_true, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, samples_random_seed,
    )

    mean_sklearn, covar_sklearn = solve_with_sklearn(observations, latent_dim, models_random_seed)

    mean_online_gradients, covar_online_gradients = solve_with_online_gradients(
        observations, latent_dim, init_factors_noise_std, gradient_optimiser_kwargs, models_random_seed,
    )

    mean_online_em, covar_online_em = solve_with_online_em(
        observations, latent_dim, init_factors_noise_std, models_random_seed,
    )

    results['covar_norm'] = torch.linalg.norm(covar_true).item()
    results['covar_distance_sklearn'] = torch.linalg.norm(covar_true - covar_sklearn).item()
    results['covar_distance_online_gradients'] = torch.linalg.norm(covar_true - covar_online_gradients).item()
    results['covar_distance_online_em'] = torch.linalg.norm(covar_true - covar_online_em).item()

    results['ll'] = compute_gaussian_log_likelihood(mean_true, covar_true, observations).item()
    results['ll_sklearn'] = compute_gaussian_log_likelihood(mean_sklearn, covar_sklearn, observations).item()
    results['ll_online_gradients'] = compute_gaussian_log_likelihood(
        mean_online_gradients, covar_online_gradients, observations,
    ).item()
    results['ll_online_em'] = compute_gaussian_log_likelihood(mean_online_em, covar_online_em, observations).item()

    return results


def generate_and_sample_fa_model(observation_dim: int, latent_dim: int, spectrum_range: List[int], n_samples: int,
                                 random_seed: int) -> (Tensor, Tensor, Tensor):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    c, F, psi = generate_fa_model(observation_dim, latent_dim, spectrum_range)
    covar = compute_covariance(F, psi)
    observations = sample_observations(c, F, psi, n_samples)
    return c, covar, observations


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
    h = p_h.sample((n_samples,))
    noise = p_psi.sample((n_samples,))
    return h.mm(F.t()) + c.reshape(1, -1) + noise


def solve_with_sklearn(observations: Tensor, latent_dim: int, random_seed: int) -> (Tensor, Tensor):
    print('Learning FA model via sklearn...')
    fa = FactorAnalysis(n_components=latent_dim, svd_method='randomized', random_state=random_seed)
    fa.fit(observations.numpy())
    mean = torch.from_numpy(fa.mean_)
    covar = torch.from_numpy(fa.get_covariance())
    return mean, covar


def solve_with_online_gradients(observations: Tensor, latent_dim: int, init_factors_noise_std: float,
                                optimiser_kwargs: dict, random_seed: int) -> (Tensor, Tensor):
    print('Learning FA model via online gradients...')
    observation_dim = observations.shape[1]
    fa = OnlineGradientFactorAnalysis(
        observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std, optimiser=SGD,
        optimiser_kwargs=optimiser_kwargs, random_seed=random_seed,
    )
    return solve_with_online_fa(fa, observations)


def solve_with_online_em(observations: Tensor, latent_dim: int, init_factors_noise_std: float, random_seed: int,
                         ) -> (Tensor, Tensor):
    print('Learning FA model via online EM...')
    observation_dim = observations.shape[1]
    fa = OnlineEMFactorAnalysis(
        observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std, random_seed=random_seed,
    )
    return solve_with_online_fa(fa, observations)


def solve_with_online_fa(fa: OnlineFactorAnalysis, observations: Tensor) -> (Tensor, Tensor):
    for theta in observations:
        fa.update(theta)
    mean = fa.c.squeeze()
    covar = compute_covariance(fa.F, torch.diag(fa.diag_psi.squeeze()))
    return mean, covar


def compute_covariance(F: Tensor, psi: Tensor) -> Tensor:
    return F.mm(F.t()) + psi


def compute_gaussian_log_likelihood(mean: Tensor, covar: Tensor, X: Tensor) -> Tensor:
    n, d = X.shape
    inv_covar = torch.linalg.inv(covar)
    centred_X = X - mean.reshape(1, -1)
    unnormalised_log_likelihood = -0.5 * torch.sum(centred_X.mm(inv_covar) * centred_X, dim=1).mean()
    log_normalising_factor = -0.5 * (torch.logdet(covar) + d * np.log(2 * np.pi))
    return unnormalised_log_likelihood + log_normalising_factor


@click.command()
@click.option('--results-output-path', type=str, help='The file path to save the experiment results')
def main(results_output_path):
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    results = run_all_experiments(
        params['online_fa']['experiments'],
        params['online_fa']['n_trials'],
        params['online_fa']['init_factors_noise_std'],
        params['online_fa']['gradient_optimiser_kwargs'],
    )
    print(results)

    Path(os.path.dirname(results_output_path)).mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_output_path)


if __name__ == '__main__':
    main()
