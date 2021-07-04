import os
from pathlib import Path
from typing import List

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


def run_all_fa_experiments(experiments_config: List[dict], n_trials: int, init_factors_noise_std: float,
                           gradient_optimiser_kwargs: dict) -> pd.DataFrame:
    """
    Run all factor analysis (FA) experiments specified in the given configuration.

    For each trial of each experiment, generate a FA model, sample observations from the model and then with this data
    estimate the parameters of the model using three different learning algorithms: sklearn's batch SVD approach, online
    stochastic gradient ascent and online expectation maximisation (EM).

    Args:
        experiments_config: Each element of the list is a dictionary specifying the configuration of a single FA
            experiment. Must contain the following fields:
                - observation_dim: (int) The size of the observed variable space of the FA model. Note that in each
                    trial the full covariance matrix of the FA model will be constructed, of shape
                    (observation_dim, observation_dim), so this should not be too big.
                - latent_dim: (int) The size of the latent variable space of the FA model.
                - spectrum_range: ([float, float]) The end points of the "spectrum", which controls the conditioning of
                    the covariance matrix of the true FA model.
                - n_samples: (int) The number of observations sampled from the true FA model. All FA learning algorithms
                    will be fit to this data.
        n_trials: The number of trials to run for each experiment, for different random seeds.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix in the online FA learning algorithms.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch SGD optimiser used in the online gradient FA
            learning algorithm.

    Returns:
        The results of each experiment. Has len(experiments_config) * n_trials rows and the following columns:
            - observation_dim: (int) Same as above.
            - latent_dim: (int) Same as above.
            - spectrum_min: (float) The lower bound of the spectrum range.
            - spectrum_max: (float) The upper bound of the spectrum range.
            - n_samples: (int) Same as above.
            - covar_norm: (float) The Frobenius norm of the the true covariance matrix of the FA model.
            - covar_distance_sklearn: (float) The Frobenius norm of the difference between the true covariance matrix
                and the covariance matrix estimated by sklearn's `FactorAnalysis`.
            - covar_distance_online_gradient: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineGradientFactorAnalysis`.
            - covar_distance_online_em: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineEMFactorAnalysis`.
            - ll_true: (float) The log-likelihood of the true FA model, given the data sampled from the model.
            - ll_sklearn: (float) The log-likelihood of the sklearn FA model.
            - ll_online_gradient: (float) The log-likelihood of the online gradient FA model.
            - ll_online_em: (float) The log-likelihood of the online EM FA model.
            - experiment: (int) The index of the experiment.
            - trial: (int) The index of the trial within the experiment.
    """
    results = []
    for i_experiment, config in enumerate(experiments_config):
        print(f'Running experiment {i_experiment + 1} of {len(experiments_config)}...')
        print('-' * 100)
        for i_trial in range(n_trials):
            print(f'Running trial {i_trial + 1} of {n_trials}...')

            trial_results = run_fa_experiment_trial(
                config['observation_dim'],
                config['latent_dim'],
                config['spectrum_range'],
                config['n_samples'],
                init_factors_noise_std,
                gradient_optimiser_kwargs,
                samples_random_seed=i_trial,
                algorithms_random_seed=i_trial + 1,
            )

            trial_results['experiment'] = i_experiment + 1
            trial_results['trial'] = i_trial + 1
            results.append(trial_results)

            print('-' * 100)
        print('-' * 100)

    return pd.DataFrame(results)


def run_fa_experiment_trial(observation_dim: int, latent_dim: int, spectrum_range: [float, float], n_samples: int,
                            init_factors_noise_std: float, gradient_optimiser_kwargs: dict, samples_random_seed: int,
                            algorithms_random_seed: int) -> dict:
    """
    Run a factor analysis (FA) experiment trial for the given parameters.

    Generate a FA model, sample observations from the model and then with this data estimate the parameters of the model
    using three different learning algorithms: sklearn's batch SVD approach, online stochastic gradient ascent and
    online expectation maximisation (EM).

    Args:
        observation_dim: The size of the observed variable space of the FA model. Note that the full covariance matrix
            of the FA model will be constructed, of shape (observation_dim, observation_dim), so this should not be too
            big.
        latent_dim: The size of the latent variable space of the FA model.
        spectrum_range: The end points of the "spectrum", which controls the conditioning of the covariance matrix of
            the true FA model.
        n_samples: The number of observations sampled from the true FA model. All FA learning algorithms will be fit to
            this data.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix in the online FA learning algorithms.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch SGD optimiser used in the online gradient FA
            learning algorithm.
        samples_random_seed: The random seed used to construct the true FA model and generate samples from it.
        algorithms_random_seed: The random seed used in all three learning algorithms.

    Returns:
        The results of the trial. Has the following keys:
            - observation_dim: (int) Same as input.
            - latent_dim: (int) Same as input.
            - spectrum_min: (float) The lower bound of the spectrum range.
            - spectrum_max: (float) The upper bound of the spectrum range.
            - n_samples: (int) Same as input.
            - covar_norm: (float) The Frobenius norm of the the true covariance matrix of the FA model.
            - covar_distance_sklearn: (float) The Frobenius norm of the difference between the true covariance matrix
                and the covariance matrix estimated by sklearn's `FactorAnalysis`.
            - covar_distance_online_gradient: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineGradientFactorAnalysis`.
            - covar_distance_online_em: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineEMFactorAnalysis`.
            - ll_true: (float) The log-likelihood of the true FA model, given the data sampled from the model.
            - ll_sklearn: (float) The log-likelihood of the sklearn FA model.
            - ll_online_gradient: (float) The log-likelihood of the online gradient FA model.
            - ll_online_em: (float) The log-likelihood of the online EM FA model.
    """
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

    mean_sklearn, covar_sklearn = learn_fa_with_sklearn(observations, latent_dim, algorithms_random_seed)

    mean_online_gradient, covar_online_gradient = learn_fa_with_online_gradients(
        observations, latent_dim, init_factors_noise_std, gradient_optimiser_kwargs, algorithms_random_seed,
    )

    mean_online_em, covar_online_em = learn_fa_with_online_em(
        observations, latent_dim, init_factors_noise_std, algorithms_random_seed,
    )

    results['covar_norm'] = compute_distance_between_matrices(covar_true, torch.zeros_like(covar_true))
    results['covar_distance_sklearn'] = compute_distance_between_matrices(covar_true, covar_sklearn)
    results['covar_distance_online_gradient'] = compute_distance_between_matrices(covar_true, covar_online_gradient)
    results['covar_distance_online_em'] = compute_distance_between_matrices(covar_true, covar_online_em)

    results['ll_true'] = compute_gaussian_log_likelihood(mean_true, covar_true, observations)
    results['ll_sklearn'] = compute_gaussian_log_likelihood(mean_sklearn, covar_sklearn, observations)
    results['ll_online_gradient'] = compute_gaussian_log_likelihood(
        mean_online_gradient, covar_online_gradient, observations,
    )
    results['ll_online_em'] = compute_gaussian_log_likelihood(mean_online_em, covar_online_em, observations)

    return results


def generate_and_sample_fa_model(observation_dim: int, latent_dim: int, spectrum_range: [float, float], n_samples: int,
                                 random_seed: int) -> (Tensor, Tensor, Tensor):
    """
    Generate a factor analysis (FA) model and sample observations from it.

    Args:
        observation_dim: The size of the observed variable space of the FA model.
        latent_dim: The size of the latent variable space of the FA model.
        spectrum_range: The end points of the "spectrum", which controls the conditioning of the covariance matrix of
            the FA model.
        n_samples: The number of observations sampled from the FA model.
        random_seed: The random seed to use for generating the mean and covariance of the FA model and then sampling
            from it.

    Returns:
        c: The mean of the FA model. Of shape (observation_dim,).
        covar: The covariance of the FA model. Of shape (observation_dim, observation_dim).
        observations: The observations sampled from the FA model. Of shape (n_samples, observation_dim).
    """
    torch.manual_seed(random_seed)
    c, F, psi = generate_fa_model(observation_dim, latent_dim, spectrum_range)
    covar = compute_fa_covariance(F, psi)
    observations = sample_fa_observations(c, F, psi, n_samples)
    return c, covar, observations


def generate_fa_model(observation_dim: int, latent_dim: int, spectrum_range: [float, float],
                      ) -> (Tensor, Tensor, Tensor):
    """
    Generate a factor analysis (FA) model.

    The mean of the FA model is sampled from a standard normal distribution.

    The factor loading matrix is generated in such a way that it is possible to control the conditioning number of
    the resulting covariance matrix of the FA model. The steps are as follows:

        1. Generate a matrix A, of shape (observation_dim, observation_dim), by sampling from a standard normal
            distribution.
        2. Compute the positive semi-definite matrix, M = A*A^T, of shape (observation_dim, observation_dim).
        3. Compute the eigenvalue decomposition of M and keep the first latent_dim eigenvectors.
        4. Generate the spectrum, s^2, of shape (observation_dim, 1), by sampling from a uniform distribution with range
            equal to spectrum_range.
        5. Multiply the eigenvectors by s to obtain the columns of the factor loading matrix, F.

    Finally, the diagonal entries of the Gaussian noise covariance matrix are sampled from a uniform distribution with
    range [0, max(s^2)].

    This approach ensures that the variance of the observation noise is maximally as large as the largest value of
    s^2. This corresponds to an assumption that Fh is the "signal" corrupted by additive noise, where h is a latent
    variable vector.

    Args:
        observation_dim: The size of the observed variable space of the FA model.
        latent_dim: The size of the latent variable space of the FA model.
        spectrum_range: The end points of the "spectrum", which controls the conditioning of the covariance matrix of
            the FA model.

    Returns:
        c: The mean of the FA model. Of shape (observation_dim,).
        F: The factor loading matrix. Of shape (observation_dim, latent_dim).
        psi: The Gaussian noise covariance matrix. Of shape (observation_dim, observation_dim).
    """
    c = torch.randn(observation_dim)

    A = torch.randn(observation_dim, observation_dim)
    M = A.mm(A.t())
    _, V = torch.linalg.eigh(M)
    Vk = V[:, :latent_dim]
    spectrum = torch.FloatTensor(observation_dim, 1).uniform_(*spectrum_range)
    F = Vk * torch.sqrt(spectrum)

    diag_psi = torch.FloatTensor(observation_dim).uniform_(0, spectrum.max())
    psi = torch.diag(diag_psi)

    return c, F, psi


def sample_fa_observations(c: Tensor, F: Tensor, psi: Tensor, n_samples: int) -> Tensor:
    """
    Sample observations from a factor analysis (FA) model.

    Observations are of the form Fh + c + noise, where h is a latent variable vector sampled from N(0, I) and the noise
    vector is sampled from N(0, psi).

    Args:
        c: The mean of the FA model. Of shape (observation_dim,).
        F: The factor loading matrix. Of shape (observation_dim, latent_dim).
        psi: The Gaussian noise covariance matrix. Of shape (observation_dim, observation_dim).
        n_samples: The number of observations sampled from the FA model.

    Returns:
        Sampled observations. Of shape (n_samples, observation_dim).
    """
    observation_dim, latent_dim = F.shape
    p_h = MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))
    p_noise = MultivariateNormal(loc=torch.zeros(observation_dim), covariance_matrix=psi)
    H = p_h.sample((n_samples,))
    noise = p_noise.sample((n_samples,))
    return H.mm(F.t()) + c.reshape(1, -1) + noise


def learn_fa_with_sklearn(observations: Tensor, latent_dim: int, random_seed: int) -> (Tensor, Tensor):
    """
    Learn the parameters of a factor analysis (FA) model using the sklearn (randomised) SVD method.

    Args:
        observations: Sampled observations. Of shape (n_samples, observation_dim).
        latent_dim: The size of the latent variable space of the FA model.
        random_seed: The random seed used in the algorithm.

    Returns:
        mean: The learned mean of the FA model.
        covar: The learned covariance matrix of the FA model.
    """
    print('Learning FA model via sklearn (randomised) SVD method...')
    fa = FactorAnalysis(n_components=latent_dim, svd_method='randomized', random_state=random_seed)
    fa.fit(observations.numpy())
    mean = torch.from_numpy(fa.mean_)
    covar = torch.from_numpy(fa.get_covariance())
    return mean, covar


def learn_fa_with_online_gradients(observations: Tensor, latent_dim: int, init_factors_noise_std: float,
                                   optimiser_kwargs: dict, random_seed: int) -> (Tensor, Tensor):
    """
    Learn the parameters of a factor analysis (FA) model via online stochastic gradient ascent.

    Args:
        observations: Sampled observations. Of shape (n_samples, observation_dim).
        latent_dim: The size of the latent variable space of the FA model.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix.
        optimiser_kwargs: Keyword arguments for the PyTorch SGD optimiser used in the algorithm.
        random_seed: The random seed used in the algorithm.

    Returns:
        mean: The learned mean of the FA model.
        covar: The learned covariance matrix of the FA model.
    """
    print('Learning FA model via online stochastic gradient ascent...')
    observation_dim = observations.shape[1]
    fa = OnlineGradientFactorAnalysis(
        observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std, optimiser=SGD,
        optimiser_kwargs=optimiser_kwargs, random_seed=random_seed,
    )
    return learn_fa_online(fa, observations)


def learn_fa_with_online_em(observations: Tensor, latent_dim: int, init_factors_noise_std: float, random_seed: int,
                            ) -> (Tensor, Tensor):
    """
    Learn the parameters of a factor analysis (FA) model via online expectation maximisation (EM).

    Args:
        observations: Sampled observations. Of shape (n_samples, observation_dim).
        latent_dim: The size of the latent variable space of the FA model.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix.
        random_seed: The random seed used in the algorithm.

    Returns:
        mean: The learned mean of the FA model.
        covar: The learned covariance matrix of the FA model.
    """
    print('Learning FA model via online expectation maximisation...')
    observation_dim = observations.shape[1]
    fa = OnlineEMFactorAnalysis(
        observation_dim, latent_dim, init_factors_noise_std=init_factors_noise_std, random_seed=random_seed,
    )
    return learn_fa_online(fa, observations)


def learn_fa_online(fa: OnlineFactorAnalysis, observations: Tensor) -> (Tensor, Tensor):
    """
    Learn the parameters of a factor analysis (FA) model online.

    The model is updated iteratively using the given observations, one by one. A single pass is made over the data.

    Args:
        fa: An initialised online FA learning algorithm.
        observations: Sampled observations. Of shape (n_samples, observation_dim).

    Returns:
        mean: The learned mean of the FA model.
        covar: The learned covariance matrix of the FA model.
    """
    for theta in observations:
        fa.update(theta)
    mean = fa.c.squeeze()
    psi = torch.diag(fa.diag_psi.squeeze())
    covar = compute_fa_covariance(fa.F, psi)
    return mean, covar


def compute_fa_covariance(F: Tensor, psi: Tensor) -> Tensor:
    """
    Compute the covariance matrix of a factor analysis (FA) model, given the factor loading matrix and the noise
    covariance matrix.

    The covariance is F*F^T + psi.

    Args:
        F: The factor loading matrix. Of shape (observation_dim, latent_dim).
        psi: The Gaussian noise covariance matrix. Of shape (observation_dim, observation_dim).

    Returns:
        The covariance matrix of the FA model. Of shape (observation_dim, observation_dim).
    """
    return F.mm(F.t()) + psi


def compute_distance_between_matrices(A: Tensor, B: Tensor) -> float:
    """
    Compute the Frobenius norm of the difference between two matrices of the same size.

    Args:
        A: Matrix of shape (n, m).
        B: Matrix of shape (n, m).

    Returns:
        The Frobenius norm of the difference between the two matrices.
    """
    return torch.linalg.norm(A - B).item()


def compute_gaussian_log_likelihood(mean: Tensor, covar: Tensor, X: Tensor) -> float:
    """
    Compute the log-likelihood of a Gaussian distribution with the given mean and covariance matrix, given the
    observations.

    Args:
        mean: The mean of the Gaussian distribution. Of shape (observation_dim,).
        covar: The covariance of the Gaussian distribution. Of shape (observation_dim, observation_dim).
        X: The observations. Of shape (n_observations, observation_dim).

    Returns:
        The log-likelihood, averaged over the given observations.
    """
    n, d = X.shape
    inv_covar = torch.linalg.inv(covar)
    centred_X = X - mean.reshape(1, -1)
    unnormalised_log_likelihood = -0.5 * torch.sum(centred_X.mm(inv_covar) * centred_X, dim=1).mean()
    log_normalising_factor = -0.5 * (torch.logdet(covar) + d * np.log(2 * np.pi))
    return (unnormalised_log_likelihood + log_normalising_factor).item()


@click.command()
@click.option('--results-output-path', type=str, help='The parquet file path to save the experiment results')
def main(results_output_path: str):
    """
    Run the factor analysis experiments specified in the configuration and save the results.

    Args:
        results_output_path: The parquet file path to save the experiment results:
    """
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    results = run_all_fa_experiments(
        params['online_fa']['experiments'],
        params['online_fa']['n_trials'],
        params['online_fa']['init_factors_noise_std'],
        params['online_fa']['gradient_optimiser_kwargs'],
    )

    print('Results:\n')
    print(results)

    Path(os.path.dirname(results_output_path)).mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_output_path)


if __name__ == '__main__':
    main()
