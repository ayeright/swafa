import pytest
import numpy as np
import torch

from experiments.online_fa import generate_and_sample_fa_model
from experiments.utils.metrics import (
    compute_gaussian_log_likelihood,
    compute_fa_covariance,
    compute_distance_between_matrices,
    matrix_sqrt,
)


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
