import pytest
import numpy as np
import torch

from experiments.online_fa import (
    run_all_fa_experiments,
    generate_and_sample_fa_model,
    generate_fa_model,
    sample_fa_observations,
    learn_fa_with_sklearn,
    learn_fa_with_online_gradients,
    learn_fa_with_online_em,
    compute_gaussian_log_likelihood,
    compute_fa_covariance,
    compute_distance_between_matrices,
    matrix_sqrt,
)


@pytest.mark.parametrize("n_experiments", [1, 3])
@pytest.mark.parametrize("n_trials", [1, 4])
@pytest.mark.parametrize("n_sample_sizes", [1, 4])
def test_fa_results_rows_and_columns(n_experiments, n_trials, n_sample_sizes):
    config = dict(
        observation_dim=10,
        latent_dim=5,
        spectrum_range=[0, 1],
        n_samples=list((np.arange(n_sample_sizes) + 1) * 10),
    )
    experiments_config = [config] * n_experiments
    results = run_all_fa_experiments(
        experiments_config,
        n_trials,
        init_factors_noise_std=0.1,
        gradient_optimiser='sgd',
        gradient_optimiser_kwargs=dict(lr=0.01),
        gradient_warm_up_time_steps=1,
        em_warm_up_time_steps=1,
        n_test_samples=1000,
    )
    expected_columns = [
        'observation_dim',
        'latent_dim',
        'spectrum_min',
        'spectrum_max',
        'n_samples',
        'covar_norm',
        'covar_distance_sklearn',
        'covar_distance_online_gradient',
        'covar_distance_online_em',
        'll_train_true',
        'll_train_sklearn',
        'll_train_online_gradient',
        'll_train_online_em',
        'll_test_true',
        'll_test_sklearn',
        'll_test_online_gradient',
        'll_test_online_em',
        'wasserstein_sklearn',
        'wasserstein_online_gradient',
        'wasserstein_online_em',
        'experiment',
        'trial'
    ]
    actual_columns = list(results.columns)
    assert len(actual_columns) == len(expected_columns)
    assert len(np.intersect1d(actual_columns, expected_columns)) == len(actual_columns)

    assert len(results) == n_experiments * n_trials * n_sample_sizes
    for i in range(n_experiments):
        assert (results['experiment'] == i + 1).sum() == n_trials * n_sample_sizes


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
@pytest.mark.parametrize('n_samples', [10, 100])
def test_true_params_shape(observation_dim, latent_dim, spectrum_range, n_samples):
    mean, F, psi, covar, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, random_seed=0,
    )
    assert mean.shape == (observation_dim,)
    assert covar.shape == (observation_dim, observation_dim)


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
def test_model_mean_matches_sample_mean(observation_dim, latent_dim, spectrum_range):
    n_samples = 100000
    c, F, psi, covar, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, random_seed=0,
    )
    sample_mean = observations.mean(dim=0)
    normalised_distance = torch.linalg.norm(c - sample_mean) / torch.linalg.norm(c)
    assert normalised_distance < 0.01


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
def test_model_covariance_matches_sample_covariance(observation_dim, latent_dim, spectrum_range):
    n_samples = 100000
    c, F, psi, covar, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, random_seed=0,
    )
    sample_covar = torch.from_numpy(np.cov(observations.t().numpy())).float()
    normalised_distance = torch.linalg.norm(covar - sample_covar) / torch.linalg.norm(covar)
    assert normalised_distance < 0.1


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
@pytest.mark.parametrize('n_samples', [10, 100])
def test_sampled_fa_observations_shape(observation_dim, latent_dim, spectrum_range, n_samples):
    c, F, psi = generate_fa_model(observation_dim, latent_dim, spectrum_range, random_seed=0)
    observations = sample_fa_observations(c, F, psi, n_samples, random_seed=0)
    assert observations.shape == (n_samples, observation_dim)


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
@pytest.mark.parametrize('n_samples', [10, 100])
def test_sklearn_learned_params_shape(observation_dim, latent_dim, spectrum_range, n_samples):
    c, F, psi, covar, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, random_seed=0,
    )
    mean, covar = learn_fa_with_sklearn(observations, latent_dim, random_seed=0)
    assert mean.shape == (observation_dim,)
    assert covar.shape == (observation_dim, observation_dim)


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
@pytest.mark.parametrize('n_samples', [10, 100])
def test_online_gradients_learned_params_shape(observation_dim, latent_dim, spectrum_range, n_samples):
    c, F, psi, covar, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, random_seed=0,
    )
    _, mean, covar = learn_fa_with_online_gradients(
        observations, latent_dim, init_factors_noise_std=0.1, optimiser_name='sgd', optimiser_kwargs=None,
        n_warm_up_time_steps=1, random_seed=0,
    )
    assert mean.shape == (observation_dim,)
    assert covar.shape == (observation_dim, observation_dim)


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
@pytest.mark.parametrize('n_samples', [10, 100])
def test_online_em_learned_params_shape(observation_dim, latent_dim, spectrum_range, n_samples):
    c, F, psi, covar, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, random_seed=0,
    )
    _, mean, covar = learn_fa_with_online_em(
        observations, latent_dim, init_factors_noise_std=0.1, n_warm_up_time_steps=1, random_seed=0,
    )
    assert mean.shape == (observation_dim,)
    assert covar.shape == (observation_dim, observation_dim)


def test_compute_fa_covariance():
    F = torch.Tensor([
        [1, 2],
        [-3, 4],
        [-1, 2],
    ])
    psi = torch.Tensor([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
    ])
    expected_covariance = torch.Tensor([
        [6, 5, 3],
        [5, 27, 11],
        [3, 11, 8],
    ])
    actual_covariance = compute_fa_covariance(F, psi)
    assert torch.isclose(actual_covariance, expected_covariance, atol=1e-05).all()


def test_compute_matrix_norm():
    X = torch.Tensor([
        [1, 2],
        [-3, 4],
    ])
    expected_norm = np.sqrt(30)
    actual_norm = compute_distance_between_matrices(X, torch.zeros_like(X))
    assert np.isclose(actual_norm, expected_norm, atol=1e-5)


def test_compute_distance_between_matrices():
    A = torch.Tensor([
        [1, 6],
        [6, 5],
    ])
    B = torch.Tensor([
        [0, 4],
        [9, 1],
    ])
    expected_distance = np.sqrt(30)
    actual_distance = compute_distance_between_matrices(A, B)
    assert np.isclose(actual_distance, expected_distance, atol=1e-5)


@pytest.mark.parametrize("observation_dim", [10, 20])
@pytest.mark.parametrize("latent_dim", [5, 8])
@pytest.mark.parametrize("spectrum_range", [[0, 1], [0, 10]])
@pytest.mark.parametrize('n_samples', [10, 100])
def test_gaussian_log_likelihood(observation_dim, latent_dim, spectrum_range, n_samples):
    mean, F, psi, covar, observations = generate_and_sample_fa_model(
        observation_dim, latent_dim, spectrum_range, n_samples, random_seed=0,
    )
    inv_cov = torch.linalg.inv(covar)
    log_det_cov = torch.logdet(covar)
    expected_ll_observations = torch.zeros(n_samples)
    for i, x in enumerate(observations):
        centred_x = (x - mean).reshape(-1, 1)
        expected_ll_observations[i] = (
            -0.5 * centred_x.t().mm(inv_cov).mm(centred_x)
            -0.5 * log_det_cov
            -0.5 * observation_dim * np.log(2 * np.pi)
        )
    expected_ll = expected_ll_observations.mean().item()
    actual_ll = compute_gaussian_log_likelihood(mean, covar, observations)
    assert np.isclose(actual_ll, expected_ll, atol=1e-5)


@pytest.mark.parametrize("dim", [2, 5, 10, 20, 50, 100])
def test_matrix_sqrt(dim):
    torch.manual_seed(0)
    B = torch.randn(dim, dim)
    A = B.mm(B.t())
    sqrt_A = matrix_sqrt(A)
    A_rebuilt = sqrt_A.mm(sqrt_A)
    assert torch.isclose(A_rebuilt, A, atol=1e-4 * dim).all()
