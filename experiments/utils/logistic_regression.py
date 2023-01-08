from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F


def generate_model_and_data(
        n_samples: int,
        feature_covar: np.ndarray,
        weight_prior_precision: float,
        random_seed: Optional[int] = None,
) -> (Tensor, Tensor, Tensor):
    """
    Generate data from a logistic regression model.

    Input features are sampled from a zero-mean multivariate Gaussian distribution with the specified covariance.

    The weights of the logistic regression model are sampled from a zero-mean multivariate Gaussian distribution with
    the specified precision.

    Binary output labels are then sampled from the resulting model, given the features.

    Note: the model does not include a bias term.

    Args:
        n_samples: The number of samples to generate.
        feature_covar: An array of shape (n_features, n_features) specifying the desired covariance of the input
            features.
        weight_prior_precision: The desired precision of the prior distribution of the model weights.
        random_seed: Seed for reproducibility.

    Returns:
        X: Input features, of shape (n_samples, n_features).
        y: Binary output labels, of shape (n_samples,)
        theta: Model weights, of shape (n_features, 1).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    n_features = len(feature_covar)
    zeros = torch.zeros(n_features)

    p_x = MultivariateNormal(loc=zeros, covariance_matrix=torch.Tensor(feature_covar))
    X = p_x.sample((n_samples,))

    p_theta = MultivariateNormal(loc=zeros, covariance_matrix=torch.eye(n_features) / weight_prior_precision)
    theta = p_theta.sample().reshape(-1, 1)

    logits = X.mm(theta).squeeze()
    probs = torch.sigmoid(logits)
    y = torch.from_numpy(np.random.binomial(1, probs.numpy())).to(torch.float)

    return X, y, theta


def approximate_2d_posterior(
        theta_range_1: Tensor,
        theta_range_2: Tensor,
        X: Tensor,
        y: Tensor,
        weight_prior_precision: float,
        scale: bool = False,
) -> Tensor:
    """
    Approximate the posterior of a logistic regression model with two weights and no bias term over a 2D grid.

    Args:
        theta_range_1: Range of values over which to compute the posterior for the first weight, of shape (m1,).
        theta_range_2: Range of values over which to compute the posterior for the second weight, of shape (m2,)
        X: Input features, of shape (n, 2).
        y: Binary output labels, of shape (n,).
        weight_prior_precision: The precision of the prior distribution of the model weights.
        scale: Whether or not to scale the posterior so that the maximum value is 1.

    Returns:
        Approximate posterior of shape (m1, m2).
    """
    unnormalised_log_probs = torch.zeros(len(theta_range_1), len(theta_range_2))

    for i, theta_1 in enumerate(theta_range_1):
        for j, theta_2 in enumerate(theta_range_2):
            theta = torch.tensor([theta_1, theta_2])
            unnormalised_log_probs[i, j] = compute_unnormalised_log_prob(theta, X, y, weight_prior_precision)

    flattened_probs = F.softmax(torch.flatten(unnormalised_log_probs), dim=0)

    posterior = flattened_probs.reshape(*unnormalised_log_probs.shape)

    if scale:
        return posterior / posterior.max()

    return posterior


def compute_unnormalised_log_prob(theta: Tensor, X: Tensor, y: Tensor, weight_prior_precision: float) -> float:
    """
    Compute the unnormalised log posterior probability of a logistic regression model, given the data.

    The model prior is a zero-mean multivariate Gaussian distribution with the specified precision.

    Args:
        theta: Weights of the model, of shape (n_features,).
        X: Input features, of shape (n_samples, n_features).
        y: Binary output labels, of shape (n_samples,).
        weight_prior_precision: The precision of the prior distribution of the model weights.

    Returns:
        The unnormalised log posterior probability of the weights.
    """
    prior_log_prob = compute_prior_log_prob(theta, weight_prior_precision)
    log_likelihood = compute_log_likelihood(theta, X, y)

    return prior_log_prob + log_likelihood


def compute_prior_log_prob(theta: Tensor, precision: float) -> float:
    """
    Compute the log prior probability of a logistic regression model.

    The prior is a zero-mean multivariate Gaussian distribution with the specified precision.

    Args:
        theta: Weights of the model, of shape (n_features,).
        precision: The precision of the prior distribution of the model weights.

    Returns:
        The log prior probability of the weights.
    """
    n_features = len(theta)

    p_theta = MultivariateNormal(loc=torch.zeros(n_features), covariance_matrix=torch.eye(n_features) / precision)

    return p_theta.log_prob(theta).item()


def compute_log_likelihood(theta: Tensor, X: Tensor, y: Tensor) -> float:
    """
    Compute the log-likelihood of a logistic regression model given the data.

    Args:
        theta: Weights of the model, of shape (n_features,).
        X: Input features, of shape (n_samples, n_features).
        y: Binary output labels, of shape (n_samples,).

    Returns:
        The log-likelihood of the weights.
    """
    logits = X.mm(theta.unsqueeze(1)).squeeze()

    return -nn.BCEWithLogitsLoss(reduction='sum')(logits, y).item()
