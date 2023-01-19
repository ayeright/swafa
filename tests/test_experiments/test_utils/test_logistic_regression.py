import numpy as np
import pytest
import torch

from experiments.utils.logistic_regression import (
    generate_model_and_data,
    approximate_2d_posterior,
    compute_unnormalised_log_prob,
    compute_prior_log_prob,
    compute_log_likelihood,
)


def test_compute_log_likelihood():
    theta = torch.tensor([1.0, 2.0])
    X = torch.tensor([
        [2.0, 1.0],
        [3.0, 1.0],
    ])
    y = torch.tensor([0.0, 1.0])

    logits = torch.tensor([4.0, 5.0])
    z = 2 * y - 1
    expected_output = torch.log(torch.sigmoid(z * logits)).sum().item()

    actual_output = compute_log_likelihood(theta, X, y)

    assert np.isclose(actual_output, expected_output)


def test_compute_prior_log_prob():
    theta = torch.tensor([1.0, 2.0])
    precision = 0.5

    expected_output = -5 / 4 - np.log(4 * np.pi)

    actual_output = compute_prior_log_prob(theta, precision)

    assert np.isclose(actual_output, expected_output)


def test_compute_unnormalised_log_prob():
    theta = torch.tensor([1.0, -2.0])
    X = torch.tensor([
        [-2.0, 0.5],
        [2.0, -1.0],
    ])
    y = torch.tensor([1.0, 0.0])
    weight_prior_precision = 0.01

    expected_output = compute_log_likelihood(theta, X, y) + compute_prior_log_prob(theta, weight_prior_precision)

    actual_output = compute_unnormalised_log_prob(theta, X, y, weight_prior_precision)

    assert np.isclose(actual_output, expected_output)


@pytest.mark.parametrize('scale', [True, False])
def test_approximate_2d_posterior(scale):
    theta_range_1 = torch.tensor([1.0, 2.0])
    theta_range_2 = torch.tensor([3.0, 4.0])
    X = torch.tensor([
        [-2.0, 0.5],
        [2.0, -1.0],
    ])
    y = torch.tensor([1.0, 0.0])
    weight_prior_precision = 0.01

    unnormalised_log_probs = torch.tensor([
        [
            compute_unnormalised_log_prob(torch.tensor([1.0, 3.0]), X, y, weight_prior_precision),
            compute_unnormalised_log_prob(torch.tensor([1.0, 4.0]), X, y, weight_prior_precision),
        ],
        [
            compute_unnormalised_log_prob(torch.tensor([2.0, 3.0]), X, y, weight_prior_precision),
            compute_unnormalised_log_prob(torch.tensor([2.0, 4.0]), X, y, weight_prior_precision),
        ],
    ])

    expected_output = torch.exp(unnormalised_log_probs) / torch.exp(unnormalised_log_probs).sum()

    if scale:
        expected_output = expected_output / expected_output.max()

    actual_output = approximate_2d_posterior(theta_range_1, theta_range_2, X, y, weight_prior_precision, scale)

    assert torch.isclose(actual_output, expected_output).all()


def test_generate_data():
    n_samples = 10000
    feature_covar = np.array([
        [1.0, 0.5],
        [0.5, 2.0],
    ])
    weight_prior_precision = 0.01
    random_seed = 1

    X, y, theta = generate_model_and_data(n_samples, feature_covar, weight_prior_precision, random_seed)

    X_mean = torch.mean(X, dim=0).numpy()
    X_cov = np.cov(X.numpy().T)

    assert X.shape == (n_samples, 2)
    assert np.isclose(X_mean, np.zeros(2), atol=1e-2).all()
    assert np.isclose(X_cov, feature_covar, atol=1e-1).all()

    logits = X.mm(theta).squeeze()
    probs = torch.sigmoid(logits)

    assert y.shape == (n_samples,)
    assert ((y == 0) | (y == 1)).all()
    assert torch.isclose(y.mean(), probs.mean(), atol=1e-2)


def test_generate_model():
    n_samples = 2
    feature_covar = np.array([
        [1.0, 0.5],
        [0.5, 2.0],
    ])
    weight_prior_precision = 0.1

    thetas = []
    for _ in range(10000):
        X, y, theta = generate_model_and_data(n_samples, feature_covar, weight_prior_precision)
        thetas.append(theta.t())

    thetas = torch.cat(thetas)
    theta_mean = torch.mean(thetas, dim=0).numpy()
    theta_cov = np.cov(thetas.numpy().T)

    assert np.isclose(theta_mean, np.zeros(2), atol=1e-1).all()
    assert np.isclose(theta_cov, np.eye(2) / weight_prior_precision, atol=5e-1).all()
