import numpy as np
import torch
from torch import Tensor


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


def compute_gaussian_wasserstein_distance(mean1: Tensor, covar1: Tensor, mean2: Tensor, covar2: Tensor) -> float:
    """
    Compute the 2-Wasserstein distance between two non-degenerate Gaussian distributions with respect to the Frobenius
    norm [1].

    Args:
        mean1: The mean of the first distribution. Of shape (observation_dim,).
        covar1: The covariance matrix of the first distribution. Of shape (observation_dim, observation_dim).
        mean2: The mean of the second distribution. Of shape (observation_dim,).
        covar2: The covariance matrix of the second distribution. Of shape (observation_dim, observation_dim).

    Returns:
        The 2-Wasserstein distance between the two distributions.

    References:
        [1] https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions
    """
    contribution_from_mean = compute_distance_between_matrices(mean1, mean2) ** 2
    sqrt_covar2 = matrix_sqrt(covar2)
    contribution_from_covar = torch.trace(
        covar1 + covar2 - 2 * matrix_sqrt(sqrt_covar2.mm(covar1).mm(sqrt_covar2))
    )
    return (contribution_from_mean + contribution_from_covar).item()


def matrix_sqrt(A: Tensor) -> Tensor:
    """
    Compute the square root of a symmetric positive semi-definite matrix.

    Args:
        A: Symmetric positive semi-definite matrix. Of shape (n, n).

    Returns
        Matrix B such that B.mm(B) = A. Of shape (n, n).
    """
    conditioning = 1e3 * 1.1920929e-07  # for float
    eigenvalues, eigenvectors = torch.linalg.eigh(A.float())
    above_cutoff = torch.abs(eigenvalues) > conditioning * torch.max(torch.abs(eigenvalues))
    psigma_diag = torch.sqrt(eigenvalues[above_cutoff])
    eigenvectors = eigenvectors[:, above_cutoff]
    return eigenvectors.mm(torch.diag(psigma_diag)).mm(eigenvectors.t()).type(A.dtype)


def pinball_loss(y: Tensor, y_hat: Tensor, alpha: float) -> float:
    """
    Compute the average pinball loss of the given targets and predictions.

    For more details see http://josephsalmon.eu/enseignement/UW/STAT593/QuantileRegression.pdf.

    Args:
        y: The true targets. Of shape (n,).
        y_hat: The predicted targets. Of shape (n,).
        alpha: The quantile for which to compute the loss.

    Returns:
        The average pinball loss.
    """
    return torch.maximum(alpha * (y - y_hat), (1 - alpha) * (y_hat - y)).mean().item()
