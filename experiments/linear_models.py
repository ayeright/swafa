import os
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
import pandas as pd
import click
import yaml
from sklearn.preprocessing import StandardScaler

from swafa.models import FeedForwardNet
from swafa.callbacks import WeightPosteriorCallback
from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis
from swafa.posterior import ModelPosterior
from experiments.utils.callbacks import (
    OnlinePosteriorEvaluationCallback,
    BatchFactorAnalysisPosteriorEvaluationCallback,
)
from experiments.utils.metrics import compute_distance_between_matrices
from experiments.utils.factory import OPTIMISER_FACTORY


def run_all_experiments(
        datasets: List[pd.DataFrame],
        dataset_labels: List[str],
        n_trials: int,
        model_optimiser: str,
        model_optimiser_kwargs: dict,
        n_epochs: int,
        batch_size: int,
        init_factors_noise_std: float,
        gradient_optimiser: str,
        gradient_optimiser_kwargs: dict,
        gradient_warm_up_time_steps: int,
        em_warm_up_time_steps: int,
        posterior_update_epoch_start: int,
        posterior_eval_epoch_frequency: int,
        precision_scaling_factor: float,
) -> pd.DataFrame:
    """
    Run experiments on the given datasets.

    For each dataset, train a linear model to predict the target variable via SGD. Use the model weight vectors sampled
    during SGD to estimate the posterior distribution of the weights via the sklearn batch factor analysis (FA)
    algorithm, online gradient FA and online expectation-maximisation (EM) FA. For each method, compute the distance
    between the true and estimated posterior. For each dataset, run experiments with the latent dimension of the FA
    models equal to 1 to d - 1, where d is the number of features in the dataset.

    Note: the posterior distribution depends on the reciprocal of the variance of the target variable and the precision
    of the prior on the weights. These are referred to as beta and alpha respectively. See [1] for more details on how
    they are calculated.

    Args:
        datasets: A list of datasets. Each dataset contains features and a target variable, where the target variable is
            in the final column.
        dataset_labels: A label for each of the datasets.
        n_trials: The number of trials to run for each experiment.
        model_optimiser: The name of the PyTorch optimiser used to train the linear models. Options are 'sgd' and
            'adam'.
        model_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used to train the linear models.
        n_epochs: The number of epochs for which to train the linear models.
        batch_size: The batch size to use when training the linear models.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix in the online FA learning algorithms.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient FA learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running means of the FA model in the
            online EM algorithm before updating the other parameters.
        posterior_update_epoch_start: The epoch on which to begin updating the estimated posterior distributions of the
            weights of the linear models.
        posterior_eval_epoch_frequency: The number of epochs between each evaluation of the estimated posteriors.
        precision_scaling_factor: The scaling factor used to compute the precision of the prior of the weights of the
            linear model. Full details in [1].

    Returns:
        The results of each experiment. The number of rows in the DataFrame is equal to
        sum[(n_features_in_dataset - 1) * n_trials for dataset in datasets] * n_epochs / posterior_eval_epoch_frequency.
        The DataFrame has the following columns:
            - epoch: (int) The training epoch on which the metrics were computed.
            - mean_distance_sklearn: (float) The Frobenius norm between the mean of the true posterior and the posterior
                estimated via the batch sklearn FA algorithm.
            - covar_distance_sklearn: (float) The Frobenius norm between the covariance matrix of the true posterior and
                the posterior estimated via the batch sklearn FA algorithm.
            - wasserstein_sklearn: (float) The 2-Wasserstein distance between the true posterior and the posterior
                estimated via the batch sklearn FA algorithm.
            - mean_distance_online_gradient: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online gradient FA.
            - covar_distance_online_gradient: (float) The Frobenius norm between the covariance matrix of the true
                posterior and the posterior estimated via online gradient FA.
            - wasserstein_online_gradient: (float) The 2-Wasserstein distance between the true posterior and the
                posterior estimated via online gradient FA.
            - mean_distance_online_em: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online EM FA.
            - covar_distance_online_em: (float) The Frobenius norm between the covariance matrix of the true posterior
                and the posterior estimated via online EM FA.
            - wasserstein_online_em: (float) The 2-Wasserstein distance between the true posterior and the posterior
                estimated via online EM FA.
            - latent_dim: (int) The latent dimension of the FA models.
            - trial: (int) The index of the trial within the experiment.
            - mean_norm: (float) The Frobenius norm of the mean vector of the true posterior.
            - covar_norm: (float) The Frobenius norm of the covariance matrix of the true posterior.
            - alpha: (float) The precision of the prior of the weights of the linear model.
            - beta: (float) The reciprocal of the variance of the dataset's target variable.
            - dataset: (str) The name of the dataset.
            - n_samples: (int) The number of samples in the dataset.
            - observation_dim: (int) The number of features in the dataset.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """
    results = []
    for label, dataset in zip(dataset_labels, datasets):
        print(f'Running experiments on {label} dataset...')
        print('-' * 100)

        dataset_results = run_dataset_experiments(
            dataset=dataset,
            dataset_label=label,
            n_trials=n_trials,
            model_optimiser=model_optimiser,
            model_optimiser_kwargs=model_optimiser_kwargs,
            n_epochs=n_epochs,
            batch_size=batch_size,
            init_factors_noise_std=init_factors_noise_std,
            gradient_optimiser=gradient_optimiser,
            gradient_optimiser_kwargs=gradient_optimiser_kwargs,
            gradient_warm_up_time_steps=gradient_warm_up_time_steps,
            em_warm_up_time_steps=em_warm_up_time_steps,
            posterior_update_epoch_start=posterior_update_epoch_start,
            posterior_eval_epoch_frequency=posterior_eval_epoch_frequency,
            precision_scaling_factor=precision_scaling_factor,
        )

        results.append(dataset_results)
        print('-' * 100)

    return pd.concat(results, ignore_index=True)


def run_dataset_experiments(
        dataset: pd.DataFrame,
        dataset_label: str,
        n_trials: int,
        model_optimiser: str,
        model_optimiser_kwargs: dict,
        n_epochs: int,
        batch_size: int,
        init_factors_noise_std: float,
        gradient_optimiser: str,
        gradient_optimiser_kwargs: dict,
        gradient_warm_up_time_steps: int,
        em_warm_up_time_steps: int,
        posterior_update_epoch_start: int,
        posterior_eval_epoch_frequency: int,
        precision_scaling_factor: float,
) -> pd.DataFrame:
    """
    Run experiments on the given dataset.

    Train a linear model to predict the target variable via SGD. Use the model weight vectors sampled
    during SGD to estimate the posterior distribution of the weights via the sklearn batch factor analysis (FA)
    algorithm, online gradient FA and online expectation-maximisation (EM) FA. For each method, compute the distance
    between the true and estimated posterior. Run experiments with the latent dimension of the FA models equal to 1 to
    d - 1, where d is the number of features in the dataset.

    Note: the posterior distribution depends on the reciprocal of the variance of the target variable and the precision
    of the prior on the weights. These are referred to as beta and alpha respectively. See [1] for more details on how
    they are calculated.

    Args:
        dataset: Contains features and a target variable, where the target variable is in the final column.
        dataset_label: A label for the dataset.
        n_trials: The number of trials to run for each experiment.
        model_optimiser: The name of the PyTorch optimiser used to train the linear models. Options are 'sgd' and
            'adam'.
        model_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used to train the linear models.
        n_epochs: The number of epochs for which to train the linear models.
        batch_size: The batch size to use when training the linear models.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix in the online FA learning algorithms.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient FA learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running means of the FA model in the
            online EM algorithm before updating the other parameters.
        posterior_update_epoch_start: The epoch on which to begin updating the estimated posterior distributions of the
            weights of the linear models.
        posterior_eval_epoch_frequency: The number of epochs between each evaluation of the estimated posteriors.
        precision_scaling_factor: The scaling factor used to compute the precision of the prior of the weights of the
            linear model. Full details in [1].

    Returns:
        The results of each experiment. The number of rows in the DataFrame is equal to
        (n_features_in_dataset - 1) * n_trials * n_epochs / posterior_eval_epoch_frequency.
        The DataFrame has the following columns:
            - epoch: (int) The training epoch on which the metrics were computed.
            - mean_distance_sklearn: (float) The Frobenius norm between the mean of the true posterior and the posterior
                estimated via the batch sklearn FA algorithm.
            - covar_distance_sklearn: (float) The Frobenius norm between the covariance matrix of the true posterior and
                the posterior estimated via the batch sklearn FA algorithm.
            - wasserstein_sklearn: (float) The 2-Wasserstein distance between the true posterior and the posterior
                estimated via the batch sklearn FA algorithm.
            - mean_distance_online_gradient: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online gradient FA.
            - covar_distance_online_gradient: (float) The Frobenius norm between the covariance matrix of the true
                posterior and the posterior estimated via online gradient FA.
            - wasserstein_online_gradient: (float) The 2-Wasserstein distance between the true posterior and the
                posterior estimated via online gradient FA.
            - mean_distance_online_em: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online EM FA.
            - covar_distance_online_em: (float) The Frobenius norm between the covariance matrix of the true posterior
                and the posterior estimated via online EM FA.
            - wasserstein_online_em: (float) The 2-Wasserstein distance between the true posterior and the posterior
                estimated via online EM FA.
            - latent_dim: (int) The latent dimension of the FA models.
            - trial: (int) The index of the trial within the experiment.
            - mean_norm: (float) The Frobenius norm of the mean vector of the true posterior.
            - covar_norm: (float) The Frobenius norm of the covariance matrix of the true posterior.
            - alpha: (float) The precision of the prior of the weights of the linear model.
            - beta: (float) The reciprocal of the variance of the dataset's target variable.
            - dataset: (str) The name of the dataset.
            - n_samples: (int) The number of samples in the dataset.
            - observation_dim: (int) The number of features in the dataset.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """
    X, y = get_features_and_targets(dataset)
    true_posterior_mean, true_posterior_covar, alpha, beta = compute_true_posterior(
        X, y, alpha_scaling_factor=precision_scaling_factor,
    )
    observation_dim = X.shape[1]

    results = []
    for latent_dim in range(1, observation_dim):
        print(f'Using a posterior with latent dimension equal to {latent_dim} and observation dimension equal to '
              f'{observation_dim}...')
        print('-' * 100)

        for i_trial in range(n_trials):
            print(f'Running trial {i_trial + 1} of {n_trials}...')

            trial_results = run_experiment_trial(
                X=X,
                y=y,
                true_posterior_mean=true_posterior_mean,
                true_posterior_covar=true_posterior_covar,
                weight_decay=alpha / beta,
                model_optimiser=model_optimiser,
                model_optimiser_kwargs=model_optimiser_kwargs,
                n_epochs=n_epochs,
                batch_size=batch_size,
                posterior_latent_dim=latent_dim,
                init_factors_noise_std=init_factors_noise_std,
                gradient_optimiser=gradient_optimiser,
                gradient_optimiser_kwargs=gradient_optimiser_kwargs,
                gradient_warm_up_time_steps=gradient_warm_up_time_steps,
                em_warm_up_time_steps=em_warm_up_time_steps,
                posterior_update_epoch_start=posterior_update_epoch_start,
                posterior_eval_epoch_frequency=posterior_eval_epoch_frequency,
                model_random_seed=i_trial,
                posterior_random_seed=i_trial + 1,
            )

            trial_results['latent_dim'] = latent_dim
            trial_results['trial'] = i_trial + 1

            results.append(trial_results)

            print('-' * 100)
        print('-' * 100)

    results = pd.concat(results)

    results['mean_norm'] = compute_distance_between_matrices(
        true_posterior_mean, torch.zeros_like(true_posterior_mean)
    )
    results['covar_norm'] = compute_distance_between_matrices(
        true_posterior_covar, torch.zeros_like(true_posterior_covar)
    )
    results['alpha'] = alpha
    results['beta'] = beta
    results['dataset'] = dataset_label
    results['n_samples'] = len(X)
    results['observation_dim'] = observation_dim

    return results


def run_experiment_trial(
        X: Tensor,
        y: Tensor,
        true_posterior_mean: Tensor,
        true_posterior_covar: Tensor,
        weight_decay: float,
        model_optimiser: str,
        model_optimiser_kwargs: dict,
        n_epochs: int,
        batch_size: int,
        posterior_latent_dim: int,
        init_factors_noise_std: float,
        gradient_optimiser: str,
        gradient_optimiser_kwargs: dict,
        gradient_warm_up_time_steps: int,
        em_warm_up_time_steps: int,
        posterior_update_epoch_start: int,
        posterior_eval_epoch_frequency: int,
        model_random_seed: int,
        posterior_random_seed: int,
) -> pd.DataFrame:
    """
    Run a single experiment trial on the given data with the given parameters.

    Train a linear model to predict the target variable via SGD. Use the model weight vectors sampled
    during SGD to estimate the posterior distribution of the weights via the sklearn batch factor analysis (FA)
    algorithm, online gradient FA and online expectation-maximisation (EM) FA. For each method, compute the distance
    between the true and estimated posterior.

    Args:
        X: The features. Of shape (n_samples, n_features).
        y: The targets. Of shape (n_samples,).
        true_posterior_mean: The mean of the true posterior. Of shape (n_features,).
        true_posterior_covar: The covariance matrix of the true posterior. Of shape (n_features, n_features).
        weight_decay: The L2 regularisation strength corresponding to the target variable noise and the precision of the
            prior of the weights of the linear model.
        model_optimiser: The name of the PyTorch optimiser used to train the linear model. Options are 'sgd' and 'adam'.
        model_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used to train the linear model.
        n_epochs: The number of epochs for which to train the linear model.
        batch_size: The batch size to use when training the linear model.
        posterior_latent_dim: The latent dimension of the estimated posterior distributions.
        init_factors_noise_std: The standard deviation of the noise used to initialise the off-diagonal entries of the
            factor loading matrix in the online FA learning algorithms.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient FA learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running means of the FA model in the
            online EM algorithm before updating the other parameters.
        posterior_update_epoch_start: The epoch on which to begin updating the estimated posterior distributions of the
            weights of the linear model.
        posterior_eval_epoch_frequency: The number of epochs between each evaluation of the estimated posteriors.
        model_random_seed: The random seed used to initialise the linear model.
        posterior_random_seed: The random seed used to initialise the estimated posterior distributions.

    Returns:
        The results from each evaluation epoch of the experiment. The number of rows in the DataFrame is equal to
        n_epochs / posterior_eval_epoch_frequency.
        The DataFrame has the following columns:
            - epoch: (int) The training epoch on which the metrics were computed.
            - mean_distance_sklearn: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via the batch sklearn FA algorithm.
            - covar_distance_sklearn: (float) The Frobenius norm between the covariance matrix of the true
                posterior and the posterior estimated via the batch sklearn FA algorithm.
            - wasserstein_sklearn: (float) The 2-Wasserstein distance between the true posterior and the
                posterior estimated via the batch sklearn FA algorithm.
            - mean_distance_online_gradient: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online gradient FA.
            - covar_distance_online_gradient: (float) The Frobenius norm between the covariance matrix of the true
                posterior and the posterior estimated via online gradient FA.
            - wasserstein_online_gradient: (float) The 2-Wasserstein distance between the true posterior and the
                posterior estimated via online gradient FA.
            - mean_distance_online_em: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online EM FA.
            - covar_distance_online_em: (float) The Frobenius norm between the covariance matrix of the true posterior
                and the posterior estimated via online EM FA.
            - wasserstein_online_em: (float) The 2-Wasserstein distance between the true posterior and the posterior
                estimated via online EM FA.
    """
    model_optimiser_kwargs = model_optimiser_kwargs or dict()
    model_optimiser_kwargs['weight_decay'] = weight_decay

    gradient_weight_posterior_kwargs = dict(
        latent_dim=posterior_latent_dim,
        optimiser=OPTIMISER_FACTORY[gradient_optimiser],
        optimiser_kwargs=gradient_optimiser_kwargs,
        init_factors_noise_std=init_factors_noise_std,
        n_warm_up_time_steps=gradient_warm_up_time_steps,
        random_seed=posterior_random_seed,
    )

    em_weight_posterior_kwargs = dict(
        latent_dim=posterior_latent_dim,
        init_factors_noise_std=init_factors_noise_std,
        n_warm_up_time_steps=em_warm_up_time_steps,
        random_seed=posterior_random_seed,
    )

    (
        model,
        gradient_posterior_update_callback,
        em_posterior_update_callback,
        sklearn_posterior_eval_callback,
        gradient_posterior_eval_callback,
        em_posterior_eval_callback,
    ) = build_model_and_callbacks(
        X=X,
        true_posterior_mean=true_posterior_mean,
        true_posterior_covar=true_posterior_covar,
        model_optimiser_class=OPTIMISER_FACTORY[model_optimiser],
        model_optimiser_kwargs=model_optimiser_kwargs,
        posterior_latent_dim=posterior_latent_dim,
        gradient_weight_posterior_kwargs=gradient_weight_posterior_kwargs,
        em_weight_posterior_kwargs=em_weight_posterior_kwargs,
        posterior_update_epoch_start=posterior_update_epoch_start,
        posterior_eval_epoch_frequency=posterior_eval_epoch_frequency,
        model_random_seed=model_random_seed,
    )

    callbacks = [
        gradient_posterior_update_callback,
        em_posterior_update_callback,
        sklearn_posterior_eval_callback,
        gradient_posterior_eval_callback,
        em_posterior_eval_callback,
    ]

    fit_model(
        X=X,
        y=y,
        model=model,
        callbacks=callbacks,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    return collate_callback_results(
        sklearn_posterior_eval_callback,
        gradient_posterior_eval_callback,
        em_posterior_eval_callback,
    )


def get_features_and_targets(dataset: pd.DataFrame) -> (Tensor, Tensor):
    """
    Separate the features and target variable from the given dataset.

    Scale each features by subtracting its mean and dividing by its standard deviation.

    Args:
        dataset: Contains features and a target variable, where the target variable is in the final column. Of shape
            (n_samples, n_features + 1).

    Returns:
        X: The scaled features. Of shape (n_samples, n_features).
        y: The targets. Of shape (n_samples,).
    """
    X = dataset.iloc[:, :-1].values
    X = torch.from_numpy(StandardScaler().fit_transform(X)).float()
    y = torch.from_numpy(dataset.iloc[:, -1].values).float()
    return X, y


def compute_true_posterior(X: Tensor, y: Tensor, alpha: Optional[float] = None, beta: Optional[float] = None,
                           alpha_scaling_factor: float = 0.1) -> (Tensor, Tensor, float, float):
    """
    Compute mean and covariance of the true posterior distribution of the weights of a linear model, given the data.

    Full derivation given in [1].

    Args:
        X: The features. Of shape (n_samples, n_features).
        y: The targets. Of shape (n_samples,).
        alpha: The precision of the prior of the weights of the linear model. If None, will be set automatically
            according to [1].
        beta: The reciprocal of the variance of the dataset's target variable. If None, will be computed from the
            observed data.
        alpha_scaling_factor: The factor used to compute alpha, if alpha is None.

    Returns:
        mu: The mean of the true posterior. Of shape (n_features,).
        S: The covariance matrix of the true posterior. Of shape (n_features, n_features).
        alpha: The precision of the prior of the weights of the linear model. If not None, will be same as input.
        beta: The reciprocal of the variance of the dataset's target variable. If not None, will be same as input.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """
    beta = beta or compute_beta(y)
    S, alpha = compute_true_posterior_covar(X, beta, alpha=alpha, alpha_scaling_factor=alpha_scaling_factor)
    m = compute_true_posterior_mean(X, y, beta, S)
    return m, S, alpha, beta


def compute_beta(y: Tensor) -> float:
    """
    Compute the reciprocal of the variance of the target variable.

    Args:
        y: The target variable. Of shape (n_samples,).

    Returns:
        The reciprocal of the variance of the target variable. Often known as beta.
    """
    return (1 / torch.var(y)).item()


def compute_true_posterior_covar(X: Tensor, beta: float, alpha: Optional[float] = None,
                                 alpha_scaling_factor: float = 0.1) -> (Tensor, float):
    """
    Compute the covariance of the true posterior distribution of the weights of a linear model, given the data.

    This is the inverse of

        alpha * I + beta * sum_n(X[n] * X[n]^T).

    If alpha is None, it will be set to

        alpha_scaling_factor * mean(diag(beta * sum_n(X[n] * X[n]^T))).

    Full derivation given in [1].

    Args:
        X: The features. Of shape (n_samples, n_features).
        beta: The reciprocal of the variance of the dataset's target variable.
        alpha: The precision of the prior of the weights of the linear model. If None, will be computed according to the
            equation above.
        alpha_scaling_factor: The factor used to compute alpha in the equation above. Only used if alpha is None.

    Returns:
        S: The covariance matrix of the true posterior. Of shape (n_features, n_features).
        alpha: The precision of the prior of the weights of the linear model. If not None, will be same as input.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """
    B = beta * torch.einsum('ij,ik->jk', X, X)
    alpha = alpha or alpha_scaling_factor * torch.diag(B).mean().item()
    A = alpha * torch.eye(len(B)) + B
    S = torch.linalg.inv(A)
    return S, alpha


def compute_true_posterior_mean(X: Tensor, y: Tensor, beta: float, S: Tensor) -> Tensor:
    """
    Compute the mean of the true posterior distribution of the weights of a linear model, given the data.

    Full derivation given in [1].

    Args:
        X: The features. Of shape (n_samples, n_features).
        y: The targets. Of shape (n_samples,).
        beta: The reciprocal of the variance of the dataset's target variable.
        S: The covariance matrix of the true posterior. Of shape (n_features, n_features).

    Returns:
        The mean of the true posterior. Of shape (n_features,).

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """
    b = beta * (y.reshape(-1, 1) * X).sum(dim=0, keepdims=True).t()
    return S.mm(b).squeeze()


def build_model_and_callbacks(
        X: Tensor,
        true_posterior_mean: Tensor,
        true_posterior_covar: Tensor,
        model_optimiser_class: Optimizer,
        model_optimiser_kwargs: dict,
        posterior_latent_dim: int,
        gradient_weight_posterior_kwargs: dict,
        em_weight_posterior_kwargs: dict,
        posterior_update_epoch_start: int,
        posterior_eval_epoch_frequency: int,
        model_random_seed: int,
) -> (FeedForwardNet, WeightPosteriorCallback, WeightPosteriorCallback, BatchFactorAnalysisPosteriorEvaluationCallback,
      OnlinePosteriorEvaluationCallback, OnlinePosteriorEvaluationCallback):
    """
    Build a linear model and callbacks which should be called during training to update and evaluate the weight
    posteriors.

    Args:
        X: The features. Of shape (n_samples, n_features).
        true_posterior_mean: The mean of the true posterior. Of shape (n_features,).
        true_posterior_covar: The covariance matrix of the true posterior. Of shape (n_features, n_features).
        model_optimiser_class: The class of the PyTorch optimiser used to train the linear model.
        model_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used to train the linear model.
        posterior_latent_dim: The latent dimension of the estimated posterior distributions.
        gradient_weight_posterior_kwargs: Keyword arguments for the instance of OnlineGradientFactorAnalysis used to
            estimate the posterior.
        em_weight_posterior_kwargs: Keyword arguments for the instance of OnlineEMFactorAnalysis used to estimate the
            posterior.
        posterior_update_epoch_start: The epoch on which to begin updating the estimated posterior distributions of the
            weights of the linear models.
        posterior_eval_epoch_frequency: The number of epochs between each evaluation of the estimated posteriors.
        model_random_seed: The random seed used to initialise the linear model.

    Returns:
        model: An linear model with the same dimension as the input data. Note that a bias term will NOT be added to the
            model.
        gradient_posterior_update_callback: Callbacks used to update the OnlineGradientFactorAnalysis weight posterior.
        em_posterior_update_callback: Callbacks used to update the OnlineEMFactorAnalysis weight posterior.
        sklearn_posterior_eval_callback: Callback used to evaluate the sklearn FactorAnalysis weight posterior.
        gradient_posterior_eval_callback: Callback used to evaluate the OnlineGradientFactorAnalysis weight posterior.
        em_posterior_eval_callback: Callback used to evaluate the OnlineEMFactorAnalysis weight posterior.
    """
    model = FeedForwardNet(
        input_dim=X.shape[1],
        bias=False,
        optimiser_class=model_optimiser_class,
        optimiser_kwargs=model_optimiser_kwargs,
        random_seed=model_random_seed,
    )

    gradient_posterior = ModelPosterior(
        model=model,
        weight_posterior_class=OnlineGradientFactorAnalysis,
        weight_posterior_kwargs=gradient_weight_posterior_kwargs,
    )

    em_posterior = ModelPosterior(
        model=model,
        weight_posterior_class=OnlineEMFactorAnalysis,
        weight_posterior_kwargs=em_weight_posterior_kwargs,
    )

    gradient_posterior_update_callback = WeightPosteriorCallback(
        posterior=gradient_posterior.weight_posterior,
        update_epoch_start=posterior_update_epoch_start,
    )

    em_posterior_update_callback = WeightPosteriorCallback(
        posterior=em_posterior.weight_posterior,
        update_epoch_start=posterior_update_epoch_start,
    )

    sklearn_posterior_eval_callback = BatchFactorAnalysisPosteriorEvaluationCallback(
        latent_dim=posterior_latent_dim,
        true_mean=true_posterior_mean,
        true_covar=true_posterior_covar,
        collect_epoch_start=posterior_update_epoch_start,
        eval_epoch_frequency=posterior_eval_epoch_frequency,
        random_seed=model_random_seed,
    )

    gradient_posterior_eval_callback = OnlinePosteriorEvaluationCallback(
        posterior=gradient_posterior.weight_posterior,
        true_mean=true_posterior_mean,
        true_covar=true_posterior_covar,
        eval_epoch_frequency=posterior_eval_epoch_frequency,
    )

    em_posterior_eval_callback = OnlinePosteriorEvaluationCallback(
        posterior=em_posterior.weight_posterior,
        true_mean=true_posterior_mean,
        true_covar=true_posterior_covar,
        eval_epoch_frequency=posterior_eval_epoch_frequency,
    )

    return (
        model,
        gradient_posterior_update_callback,
        em_posterior_update_callback,
        sklearn_posterior_eval_callback,
        gradient_posterior_eval_callback,
        em_posterior_eval_callback,
    )


def fit_model(X: Tensor, y: Tensor, model: FeedForwardNet, callbacks: List[Callback], n_epochs: int, batch_size: int):
    """
    Fit the given model on the given data.

    Args:
        X: The features. Of shape (n_samples, n_features).
        y: The targets. Of shape (n_samples,).
        model: The model which is to be fit to the data.
        callbacks: Any callbacks which should be called during training.
        n_epochs: The number of epochs for which to train the model.
        batch_size: The batch size to use when training the model.
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    trainer = Trainer(max_epochs=n_epochs, callbacks=callbacks, progress_bar_refresh_rate=0)
    trainer.fit(model, train_dataloader=dataloader)


def collate_callback_results(sklearn_posterior_eval_callback: BatchFactorAnalysisPosteriorEvaluationCallback,
                             gradient_posterior_eval_callback: OnlinePosteriorEvaluationCallback,
                             em_posterior_eval_callback: OnlinePosteriorEvaluationCallback) -> pd.DataFrame:
    """
    Collate the results from the posterior evaluations callbacks into a single DataFrame.

    Args:
        sklearn_posterior_eval_callback: Callback used to evaluate the sklearn FactorAnalysis weight posterior.
        gradient_posterior_eval_callback: Callback used to evaluate the OnlineGradientFactorAnalysis weight posterior.
        em_posterior_eval_callback: Callback used to evaluate the OnlineEMFactorAnalysis weight posterior.

    Returns:
        The callback results from each evaluation epoch. The number of rows in the DataFrame is equal to
        n_epochs / posterior_eval_epoch_frequency.
        The DataFrame has the following columns:
            - epoch: (int) The training epoch on which the metrics were computed.
            - mean_distance_sklearn: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via the batch sklearn FA algorithm.
            - covar_distance_sklearn: (float) The Frobenius norm between the covariance matrix of the true
                posterior and the posterior estimated via the batch sklearn FA algorithm.
            - wasserstein_sklearn: (float) The 2-Wasserstein distance between the true posterior and the
                posterior estimated via the batch sklearn FA algorithm.
            - mean_distance_online_gradient: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online gradient FA.
            - covar_distance_online_gradient: (float) The Frobenius norm between the covariance matrix of the true
                posterior and the posterior estimated via online gradient FA.
            - wasserstein_online_gradient: (float) The 2-Wasserstein distance between the true posterior and the
                posterior estimated via online gradient FA.
            - mean_distance_online_em: (float) The Frobenius norm between the mean of the true posterior and the
                posterior estimated via online EM FA.
            - covar_distance_online_em: (float) The Frobenius norm between the covariance matrix of the true posterior
                and the posterior estimated via online EM FA.
            - wasserstein_online_em: (float) The 2-Wasserstein distance between the true posterior and the posterior
                estimated via online EM FA.
    """
    results = []
    for i, (epoch_sklearn, epoch_gradient, epoch_em) in enumerate(zip(sklearn_posterior_eval_callback.eval_epochs,
                                                                      gradient_posterior_eval_callback.eval_epochs,
                                                                      em_posterior_eval_callback.eval_epochs)):
        if (epoch_sklearn != epoch_gradient) or (epoch_sklearn != epoch_gradient):
            raise RuntimeError(f'The evaluation epochs of the three evaluation callbacks must be equal, not '
                               f'{epoch_sklearn}, {epoch_gradient} and {epoch_em}')

        results.append(dict(
            epoch=epoch_sklearn,
            mean_distance_sklearn=sklearn_posterior_eval_callback.distances_from_mean[i],
            covar_distance_sklearn=sklearn_posterior_eval_callback.distances_from_covar[i],
            wasserstein_sklearn=sklearn_posterior_eval_callback.wasserstein_distances[i],
            mean_distance_online_gradient=gradient_posterior_eval_callback.distances_from_mean[i],
            covar_distance_online_gradient=gradient_posterior_eval_callback.distances_from_covar[i],
            wasserstein_online_gradient=gradient_posterior_eval_callback.wasserstein_distances[i],
            mean_distance_online_em=em_posterior_eval_callback.distances_from_mean[i],
            covar_distance_online_em=em_posterior_eval_callback.distances_from_covar[i],
            wasserstein_online_em=em_posterior_eval_callback.wasserstein_distances[i],
        ))

    return pd.DataFrame(results)


@click.command()
@click.option('--boston-housing-input-path', type=str, help='The parquet file path to load the Boston Housing dataset')
@click.option('--yacht-hydrodynamics-input-path', type=str, help='The parquet file path to load the Yacht '
                                                                 'Hydrodynamics dataset')
@click.option('--concrete-strength-input-path', type=str, help='The parquet file path to load the Concrete '
                                                               'Compressive Strength dataset')
@click.option('--energy-efficiency-input-path', type=str, help='The parquet file path to load the Energy Efficiency '
                                                               'dataset')
@click.option('--results-output-path', type=str, help='The parquet file path to save the experiment results')
def main(boston_housing_input_path: str, yacht_hydrodynamics_input_path: str, concrete_strength_input_path: str,
         energy_efficiency_input_path: str, results_output_path: str):
    """
    Run experiments involving linear models on UCI datasets.

    Args:
        boston_housing_input_path: The parquet file path to load the Boston Housing dataset.
        yacht_hydrodynamics_input_path: The parquet file path to load the Yacht Hydrodynamics dataset.
        concrete_strength_input_path: The parquet file path to load the Concrete Compressive Strength dataset.
        energy_efficiency_input_path: The parquet file path to load the Energy Efficiency dataset.
        results_output_path: The parquet file path to save the experiment results.
    """
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)['linear_models']

    datasets = [
        pd.read_parquet(boston_housing_input_path),
        pd.read_parquet(yacht_hydrodynamics_input_path),
        pd.read_parquet(concrete_strength_input_path),
        pd.read_parquet(energy_efficiency_input_path),
    ]

    dataset_labels = [
        'boston_housing',
        'yacht_hydrodynamics',
        'concrete_strength',
        'energy_efficiency',
    ]

    results = run_all_experiments(
        datasets=datasets,
        dataset_labels=dataset_labels,
        n_trials=params['n_trials'],
        model_optimiser=params['model_optimiser'],
        model_optimiser_kwargs=params['model_optimiser_kwargs'],
        n_epochs=params['n_epochs'],
        batch_size=params['batch_size'],
        init_factors_noise_std=params['init_factors_noise_std'],
        gradient_optimiser=params['gradient_optimiser'],
        gradient_optimiser_kwargs=params['gradient_optimiser_kwargs'],
        gradient_warm_up_time_steps=params['gradient_warm_up_time_steps'],
        em_warm_up_time_steps=params['em_warm_up_time_steps'],
        posterior_update_epoch_start=params['posterior_update_epoch_start'],
        posterior_eval_epoch_frequency=params['posterior_eval_epoch_frequency'],
        precision_scaling_factor=params['precision_scaling_factor'],
    )

    print('Results:\n')
    print(results)

    Path(os.path.dirname(results_output_path)).mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_output_path)


if __name__ == '__main__':
    main()
