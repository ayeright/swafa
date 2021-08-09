import os
from pathlib import Path
import random
from typing import List, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.decomposition import FactorAnalysis
import yaml
import click

from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis, OnlineFactorAnalysis
from experiments.utils.factory import OPTIMISER_FACTORY
from experiments.utils.metrics import (
    compute_fa_covariance,
    compute_distance_between_matrices,
    compute_gaussian_log_likelihood,
    compute_gaussian_wasserstein_distance,
)


def run_all_fa_experiments(experiments_config: List[dict], n_trials: int, init_factors_noise_std: float,
                           gradient_optimiser: str, gradient_optimiser_kwargs: dict,
                           gradient_warm_up_time_steps: int, em_warm_up_time_steps: int, n_test_samples: int,
                           ) -> pd.DataFrame:
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
                - n_samples: (List[int]) The number of observations sampled from the true FA model. All FA learning
                    algorithms will be fit to this data. In the case of the batch algorithm, a separate model will be
                    fit to each different sample size in the list. In the case of the online algorithms, training with
                    n_samples[i] observations will begin from the model fit to n_samples[i - 1] observations. Sample
                    sizes must come in increasing order.
        n_trials: The number of trials to run for each experiment, for different random seeds.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix in the online FA learning algorithms.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online EM algorithm before updating the other parameters.
        n_test_samples: The number of observations to sample for a hold-out set to compute the test log-likelihood of
            the models.

    Returns:
        The results of each experiment. The number of rows in the DataFrame is equal to
        sum[len(config[n_samples]) * n_trials for config in experiments_config].
        The DataFrame has the following columns:
            - observation_dim: (int) Same as above.
            - latent_dim: (int) Same as above.
            - spectrum_min: (float) The lower bound of the spectrum range.
            - spectrum_max: (float) The upper bound of the spectrum range.
            - n_samples: (int) The number of samples used to fit the model.
            - covar_norm: (float) The Frobenius norm of the the true covariance matrix of the FA model.
            - covar_distance_sklearn: (float) The Frobenius norm of the difference between the true covariance matrix
                and the covariance matrix estimated by sklearn's `FactorAnalysis`.
            - covar_distance_online_gradient: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineGradientFactorAnalysis`.
            - covar_distance_online_em: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineEMFactorAnalysis`.
            - ll_train_true: (float) The training log-likelihood of the true FA model.
            - ll_train_sklearn: (float) The training log-likelihood of the sklearn FA model.
            - ll_train_online_gradient: (float) The training log-likelihood of the online gradient FA model.
            - ll_train_online_em: (float) The training log-likelihood of the online EM FA model.
            - ll_test_true: (float) The hold-out log-likelihood of the true FA model.
            - ll_test_sklearn: (float) The hold-out log-likelihood of the sklearn FA model.
            - ll_test_online_gradient: (float) The hold-out log-likelihood of the online gradient FA model.
            - ll_test_online_em: (float) The hold-out log-likelihood of the online EM FA model.
            - wasserstein_sklearn: (float) The Wasserstein distance between the Gaussian distribution defined by the
                true FA model and the Gaussian distribution defined by the sklearn FA model.
            - wasserstein_online_gradient: (float) The Wasserstein distance between the Gaussian distribution defined by
                the true FA model and the Gaussian distribution defined by the online gradient FA model.
            - wasserstein_online_em: (float) The Wasserstein distance between the Gaussian distribution defined by the
                true FA model and the Gaussian distribution defined by the online EM FA model.
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
                observation_dim=config['observation_dim'],
                latent_dim=config['latent_dim'],
                spectrum_range=config['spectrum_range'],
                n_samples=config['n_samples'],
                init_factors_noise_std=init_factors_noise_std,
                gradient_optimiser=gradient_optimiser,
                gradient_optimiser_kwargs=gradient_optimiser_kwargs,
                gradient_warm_up_time_steps=gradient_warm_up_time_steps,
                em_warm_up_time_steps=em_warm_up_time_steps,
                n_test_samples=n_test_samples,
                samples_random_seed=i_trial,
                algorithms_random_seed=i_trial + 1,
            )

            trial_results['experiment'] = i_experiment + 1
            trial_results['trial'] = i_trial + 1
            results.append(trial_results)

            print('-' * 100)
        print('-' * 100)

    return pd.concat(results)


def run_fa_experiment_trial(observation_dim: int, latent_dim: int, spectrum_range: [float, float], n_samples: List[int],
                            init_factors_noise_std: float, gradient_optimiser: str, gradient_optimiser_kwargs: dict,
                            gradient_warm_up_time_steps: int, em_warm_up_time_steps: int, n_test_samples: int,
                            samples_random_seed: int, algorithms_random_seed: int) -> pd.DataFrame:
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
            this data. In the case of the batch algorithm, a separate model will be fit to each different sample size in
            the list. In the case of the online algorithms, training with n_samples[i] observations will begin from the
            model fit to n_samples[i - 1] observations. Sample sizes must come in increasing order.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix in the online FA learning algorithms.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online EM algorithm before updating the other parameters.
        n_test_samples: The number of observations to sample for a hold-out set to compute the test log-likelihood of
            the models.
        samples_random_seed: The random seed used to construct the true FA model and generate samples from it.
        algorithms_random_seed: The random seed used in all three learning algorithms.

    Returns:
        The results of each trial. Has len(n_samples) rows and the following columns:
            - observation_dim: (int) Same as input.
            - latent_dim: (int) Same as input.
            - spectrum_min: (float) The lower bound of the spectrum range.
            - spectrum_max: (float) The upper bound of the spectrum range.
            - n_samples: (int) The number of samples used to fit the models.
            - covar_norm: (float) The Frobenius norm of the the true covariance matrix of the FA model.
            - covar_distance_sklearn: (float) The Frobenius norm of the difference between the true covariance matrix
                and the covariance matrix estimated by sklearn's `FactorAnalysis`.
            - covar_distance_online_gradient: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineGradientFactorAnalysis`.
            - covar_distance_online_em: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineEMFactorAnalysis`.
            - ll_train_true: (float) The training log-likelihood of the true FA model.
            - ll_train_sklearn: (float) The training log-likelihood of the sklearn FA model.
            - ll_train_online_gradient: (float) The training log-likelihood of the online gradient FA model.
            - ll_train_online_em: (float) The training log-likelihood of the online EM FA model.
            - ll_test_true: (float) The hold-out log-likelihood of the true FA model.
            - ll_test_sklearn: (float) The hold-out log-likelihood of the sklearn FA model.
            - ll_test_online_gradient: (float) The hold-out log-likelihood of the online gradient FA model.
            - ll_test_online_em: (float) The hold-out log-likelihood of the online EM FA model.
            - wasserstein_sklearn: (float) The Wasserstein distance between the Gaussian distribution defined by the
                true FA model and the Gaussian distribution defined by the sklearn FA model.
            - wasserstein_online_gradient: (float) The Wasserstein distance between the Gaussian distribution defined by
                the true FA model and the Gaussian distribution defined by the online gradient FA model.
            - wasserstein_online_em: (float) The Wasserstein distance between the Gaussian distribution defined by the
                true FA model and the Gaussian distribution defined by the online EM FA model.
    """
    max_samples = n_samples[-1]
    mean_true, F_true, psi_true, covar_true, observations_train = generate_and_sample_fa_model(
        observation_dim=observation_dim,
        latent_dim=latent_dim,
        spectrum_range=spectrum_range,
        n_samples=max_samples,
        random_seed=samples_random_seed,
    )

    observations_test = sample_fa_observations(
        c=mean_true,
        F=F_true,
        psi=psi_true,
        n_samples=n_test_samples,
        random_seed=random.randint(0, 1000000),
    )

    covar_norm = compute_distance_between_matrices(covar_true, torch.zeros_like(covar_true))
    ll_train_true = compute_gaussian_log_likelihood(mean_true, covar_true, observations_train)
    ll_test_true = compute_gaussian_log_likelihood(mean_true, covar_true, observations_test)

    fa_online_gradient = None
    fa_online_em = None
    samples = Tensor([])
    n_samples_iterator = [0] + n_samples

    all_results = []
    for i, n_samples_first in enumerate(n_samples_iterator[:-1]):
        n_samples_last = n_samples_iterator[i + 1]
        new_samples = observations_train[n_samples_first:n_samples_last, :]
        samples = torch.cat([samples, new_samples])

        print(f'Using {len(samples)} samples...')

        mean_sklearn, covar_sklearn = learn_fa_with_sklearn(
            observations=samples,
            latent_dim=latent_dim,
            random_seed=algorithms_random_seed,
        )

        fa_online_gradient, mean_online_gradient, covar_online_gradient = learn_fa_with_online_gradients(
            observations=new_samples,
            latent_dim=latent_dim,
            init_factors_noise_std=init_factors_noise_std,
            optimiser_name=gradient_optimiser,
            optimiser_kwargs=gradient_optimiser_kwargs,
            n_warm_up_time_steps=gradient_warm_up_time_steps,
            random_seed=algorithms_random_seed,
            fa=fa_online_gradient,
        )

        fa_online_em, mean_online_em, covar_online_em = learn_fa_with_online_em(
            observations=new_samples,
            latent_dim=latent_dim,
            init_factors_noise_std=init_factors_noise_std,
            n_warm_up_time_steps=em_warm_up_time_steps,
            random_seed=algorithms_random_seed,
            fa=fa_online_em,
        )

        print('Computing metrics...')

        results = dict(
            observation_dim=observation_dim,
            latent_dim=latent_dim,
            spectrum_min=spectrum_range[0],
            spectrum_max=spectrum_range[1],
            n_samples=n_samples_last,
            covar_norm=covar_norm,
            covar_distance_sklearn=compute_distance_between_matrices(covar_true, covar_sklearn),
            covar_distance_online_gradient=compute_distance_between_matrices(covar_true, covar_online_gradient),
            covar_distance_online_em=compute_distance_between_matrices(covar_true, covar_online_em),
            ll_train_true=ll_train_true,
            ll_train_sklearn=compute_gaussian_log_likelihood(mean_sklearn, covar_sklearn, observations_train),
            ll_train_online_gradient=compute_gaussian_log_likelihood(
                mean_online_gradient, covar_online_gradient, observations_train,
            ),
            ll_train_online_em=compute_gaussian_log_likelihood(mean_online_em, covar_online_em, observations_train),
            ll_test_true=ll_test_true,
            ll_test_sklearn=compute_gaussian_log_likelihood(mean_sklearn, covar_sklearn, observations_test),
            ll_test_online_gradient=compute_gaussian_log_likelihood(
                mean_online_gradient, covar_online_gradient, observations_test,
            ),
            ll_test_online_em=compute_gaussian_log_likelihood(mean_online_em, covar_online_em, observations_test),
            wasserstein_sklearn=compute_gaussian_wasserstein_distance(
                mean_true, covar_true, mean_sklearn, covar_sklearn,
            ),
            wasserstein_online_gradient=compute_gaussian_wasserstein_distance(
                mean_true, covar_true, mean_online_gradient, covar_online_gradient,
            ),
            wasserstein_online_em=compute_gaussian_wasserstein_distance(
                mean_true, covar_true, mean_online_em, covar_online_em,
            ),
        )

        all_results.append(results)

    return pd.DataFrame(all_results)


def generate_and_sample_fa_model(observation_dim: int, latent_dim: int, spectrum_range: [float, float], n_samples: int,
                                 random_seed: int) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
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
        F: The factor loading matrix. Of shape (observation_dim, latent_dim).
        psi: The Gaussian noise covariance matrix. Of shape (observation_dim, observation_dim).
        covar: The full covariance of the FA model. Of shape (observation_dim, observation_dim).
        observations: The observations sampled from the FA model. Of shape (n_samples, observation_dim).
    """
    print('Generating and sampling FA model...')
    c, F, psi = generate_fa_model(observation_dim, latent_dim, spectrum_range, random_seed)
    covar = compute_fa_covariance(F, psi)
    observations = sample_fa_observations(c, F, psi, n_samples, random_seed)
    return c, F, psi, covar, observations


def generate_fa_model(observation_dim: int, latent_dim: int, spectrum_range: [float, float], random_seed: int,
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
        random_seed: The random seed to use for generating the mean and covariance of the FA model.

    Returns:
        c: The mean of the FA model. Of shape (observation_dim,).
        F: The factor loading matrix. Of shape (observation_dim, latent_dim).
        psi: The Gaussian noise covariance matrix. Of shape (observation_dim, observation_dim).
    """
    torch.manual_seed(random_seed)

    c = torch.randn(observation_dim)

    A = torch.randn(observation_dim, observation_dim)
    M = A.mm(A.t())
    _, V = torch.linalg.eigh(M)
    Vk = V[:, -latent_dim:].fliplr()  # torch.linalg.eigh returns eigenvalues in ascending order
    spectrum = torch.FloatTensor(observation_dim, 1).uniform_(*spectrum_range)
    F = Vk * torch.sqrt(spectrum)

    diag_psi = torch.FloatTensor(observation_dim).uniform_(0, spectrum.max())
    psi = torch.diag(diag_psi)

    return c, F, psi


def sample_fa_observations(c: Tensor, F: Tensor, psi: Tensor, n_samples: int, random_seed: int) -> Tensor:
    """
    Sample observations from a factor analysis (FA) model.

    Observations are of the form Fh + c + noise, where h is a latent variable vector sampled from N(0, I) and the noise
    vector is sampled from N(0, psi).

    Args:
        c: The mean of the FA model. Of shape (observation_dim,).
        F: The factor loading matrix. Of shape (observation_dim, latent_dim).
        psi: The Gaussian noise covariance matrix. Of shape (observation_dim, observation_dim).
        n_samples: The number of observations sampled from the FA model.
        random_seed: The random seed to use for sampling the observations.

    Returns:
        Sampled observations. Of shape (n_samples, observation_dim).
    """
    torch.manual_seed(random_seed)

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
    mean = torch.from_numpy(fa.mean_).float()
    covar = torch.from_numpy(fa.get_covariance()).float()
    return mean, covar


def learn_fa_with_online_gradients(observations: Tensor, latent_dim: int, init_factors_noise_std: float,
                                   optimiser_name: str, optimiser_kwargs: dict, n_warm_up_time_steps: int,
                                   random_seed: int, fa: Optional[OnlineGradientFactorAnalysis] = None,
                                   ) -> (Tensor, Tensor):
    """
    Learn the parameters of a factor analysis (FA) model via online stochastic gradient ascent.

    Args:
        observations: Sampled observations. Of shape (n_samples, observation_dim).
        latent_dim: The size of the latent variable space of the FA model.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix.
        optimiser_name: The name of the PyTorch optimiser used in the learning algorithm. Options are 'sgd' and 'adam'.
        optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the learning algorithm.
        n_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model before
            updating the other parameters.
        random_seed: The random seed used in the algorithm.
        fa: If a FA model is provided, the observations will be used to fit this model. Else a completely new model will
            be initialised.

    Returns:
        fa: The trained FA model.
        mean: The learned mean of the FA model.
        covar: The learned covariance matrix of the FA model.
    """
    print('Learning FA model via online stochastic gradient ascent...')
    if fa is None:
        observation_dim = observations.shape[1]
        optimiser = OPTIMISER_FACTORY[optimiser_name]
        fa = OnlineGradientFactorAnalysis(
            observation_dim=observation_dim,
            latent_dim=latent_dim,
            init_factors_noise_std=init_factors_noise_std,
            optimiser=optimiser,
            optimiser_kwargs=optimiser_kwargs,
            n_warm_up_time_steps=n_warm_up_time_steps,
            random_seed=random_seed,
        )
    return learn_fa_online(fa, observations)


def learn_fa_with_online_em(observations: Tensor, latent_dim: int, init_factors_noise_std: float,
                            n_warm_up_time_steps: int, random_seed: int, fa: Optional[OnlineEMFactorAnalysis] = None,
                            ) -> (Tensor, Tensor):
    """
    Learn the parameters of a factor analysis (FA) model via online expectation maximisation (EM).

    Args:
        observations: Sampled observations. Of shape (n_samples, observation_dim).
        latent_dim: The size of the latent variable space of the FA model.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix.
        n_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model before
            updating the other parameters.
        random_seed: The random seed used in the algorithm.
        fa: If a FA model is provided, the observations will be used to fit this model. Else a completely new model will
            be initialised.

    Returns:
        fa: The trained FA model.
        mean: The learned mean of the FA model.
        covar: The learned covariance matrix of the FA model.
    """
    print('Learning FA model via online expectation maximisation...')
    if fa is None:
        observation_dim = observations.shape[1]
        fa = OnlineEMFactorAnalysis(
            observation_dim=observation_dim,
            latent_dim=latent_dim,
            init_factors_noise_std=init_factors_noise_std,
            n_warm_up_time_steps=n_warm_up_time_steps,
            random_seed=random_seed,
        )
    return learn_fa_online(fa, observations)


def learn_fa_online(fa: OnlineFactorAnalysis, observations: Tensor) -> (OnlineFactorAnalysis, Tensor, Tensor):
    """
    Learn the parameters of a factor analysis (FA) model online.

    The model is updated iteratively using the given observations, one by one. A single pass is made over the data.

    Args:
        fa: An initialised online FA learning algorithm.
        observations: Sampled observations. Of shape (n_samples, observation_dim).

    Returns:
        fa: The trained FA model.
        mean: The learned mean of the FA model.
        covar: The learned covariance matrix of the FA model.
    """
    for theta in observations:
        fa.update(theta)
    mean = fa.c.squeeze()
    covar = fa.get_covariance()
    return fa, mean, covar


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
        experiments_config=params['online_fa']['experiments'],
        n_trials=params['online_fa']['n_trials'],
        init_factors_noise_std=params['online_fa']['init_factors_noise_std'],
        gradient_optimiser=params['online_fa']['gradient_optimiser'],
        gradient_optimiser_kwargs=params['online_fa']['gradient_optimiser_kwargs'],
        gradient_warm_up_time_steps=params['online_fa']['gradient_warm_up_time_steps'],
        em_warm_up_time_steps=params['online_fa']['em_warm_up_time_steps'],
        n_test_samples=params['online_fa']['n_test_samples'],
    )

    print('Results:\n')
    print(results)

    Path(os.path.dirname(results_output_path)).mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_output_path)


if __name__ == '__main__':
    main()
