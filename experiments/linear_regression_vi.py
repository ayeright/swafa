import os
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
from pytorch_lightning import Trainer
import pandas as pd
import click
import yaml
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import numpy as np

from swafa.models import FeedForwardGaussianNet
from swafa.callbacks import FactorAnalysisVariationalInferenceCallback
from experiments.linear_regression_posterior import get_features_and_targets
from experiments.utils.metrics import compute_distance_between_matrices, compute_gaussian_wasserstein_distance
from experiments.utils.factory import OPTIMISER_FACTORY


def run_all_experiments(
        datasets: List[pd.DataFrame],
        dataset_labels: List[str],
        dataset_params: Dict[str, dict],
        testing: bool,
        results_output_dir: str,
) -> pd.DataFrame:
    """
    Run posterior estimation experiments on the given datasets.

    For each dataset, compute the true posterior of a linear regression model fit to the data and compare it to the
    posterior estimated via the VIFA algorithm. See [1] for more details on VIFA.

    Save all results (including plots) to the given output directory.

    Args:
        datasets: A list of datasets. Each dataset contains features and a target variable, where the target variable is
            in the final column.
        dataset_labels: A label for each of the datasets.
        dataset_params: A dict of the form label: params, where label is in dataset_labels and params is a dict
            containing the follow fields, which are parameters of the VIFA algorithm:
                latent_dim: (int) The latent dimension of the factor analysis model used as the variational
                    distribution.
                n_gradients_per_update: (int) The number of mini-batch gradients to use to form the expectation of the
                    true gradient for each parameter update.
                optimiser: (Optimizer) The class of the optimiser to use for gradient updates.
                bias_optimiser_kwargs: (dict) Keyword arguments for the optimiser which updates the bias term of the
                    factor analysis variational distribution.
                factors_optimiser_kwargs: (dict) Keyword arguments for the optimiser which updates the factor loading
                    matrix of the factor analysis variational distribution.
                noise_optimiser_kwargs: (dict) Keyword arguments for the optimiser which updates the logarithm of the
                    diagonal entries of the Gaussian noise covariance matrix of the factor analysis variational
                    distribution.
                max_grad_norm: (float) Maximum norm for gradients which are used to update the parameters of the
                    variational distribution.
                batch_size: (int) The batch size to use for mini-batch gradient optimisation.
                n_epochs: (int) The number of epochs for which to run the mini-batch gradient optimisation.
        testing: Whether or not the experiments should use the test data. Each dataset it split into train and test sets
            of equal size, and the split is always the same. If testing is True, the test data will be used, else the
            train data will be used.
        results_output_dir: The path to directory where experiment results (including plots) will be saved.

    Returns:
       A DataFrame with the following columns:
            - relative_distance_from_mean: The Frobenius norm between the mean of the true posterior and the mean of the
                variational posterior (including bias), divided by the Frobenius norm of the mean of the true
                posterior.
            - relative_distance_from_covar: The Frobenius norm between the covariance of the true posterior and the
                covariance of the variational posterior (not including bias), divided by the Frobenius norm of the
                covariance of the true posterior.
            - wasserstein_distance: The 2-Wasserstein distance between the true posterior and the variational posterior
                (not including bias).
            - alpha: The precision of the prior.
            - beta: The precision of the label noise.
            - dataset: The name of the dataset.

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """
    results = []

    for label, dataset in zip(dataset_labels, datasets):
        print(f'Running experiment on {label} dataset...')

        params = dataset_params[label]
        train_dataset, test_dataset = train_test_split(dataset)
        experiment_dataset = test_dataset if testing else train_dataset

        dataset_results = run_dataset_experiment(
            dataset=experiment_dataset,
            dataset_label=label,
            latent_dim=params['latent_dim'],
            n_gradients_per_update=params['n_gradients_per_update'],
            optimiser_class=OPTIMISER_FACTORY[params['optimiser']],
            bias_optimiser_kwargs=params['bias_optimiser_kwargs'],
            factors_optimiser_kwargs=params['factors_optimiser_kwargs'],
            noise_optimiser_kwargs=params['noise_optimiser_kwargs'],
            max_grad_norm=params['max_grad_norm'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            results_output_dir=results_output_dir,
        )

        results.append(dataset_results)

    return pd.DataFrame(results)


def train_test_split(dataset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the data into equally sized train and test sets.

    Args:
        dataset: Data of shape (n, k).

    Returns:
        train_dataset: Training data of shape (n / 2, k).
        test_dataset: Test data of shape (n / 2, k).
    """
    shuffled_dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    middle_index = int(len(shuffled_dataset) / 2)
    train_dataset = shuffled_dataset.iloc[:middle_index, :]
    test_dataset = shuffled_dataset.iloc[middle_index:, :]

    return train_dataset, test_dataset


def run_dataset_experiment(
        dataset: pd.DataFrame,
        dataset_label: str,
        latent_dim: int,
        n_gradients_per_update: int,
        optimiser_class: Optimizer,
        bias_optimiser_kwargs: dict,
        factors_optimiser_kwargs: dict,
        noise_optimiser_kwargs: dict,
        max_grad_norm: float,
        batch_size: int,
        n_epochs: int,
        results_output_dir: str,
) -> dict:
    """
    Run posterior estimation experiments on the given dataset.

    Compute the true posterior of a linear regression model fit to the data and compare it to the posterior estimated
    via the VIFA algorithm.

    Save all results (including plots) to the given output directory.

    Args:
        dataset: Contains features and a target variable, where the target variable is in the final column.
        dataset_label: A label for the dataset. Used for saving results.
        latent_dim: The latent dimension of the factor analysis model used as the variational distribution.
        n_gradients_per_update: The number of mini-batch gradients to use to form the expectation of the true gradient
            for each parameter update.
        optimiser_class: The class of the optimiser to use for gradient updates.
        bias_optimiser_kwargs: Keyword arguments for the optimiser which updates the bias term of the factor analysis
            variational distribution.
        factors_optimiser_kwargs: Keyword arguments for the optimiser which updates the factor loading matrix of the
            factor analysis variational distribution.
        noise_optimiser_kwargs: Keyword arguments for the optimiser which updates the logarithm of the diagonal entries
            of the Gaussian noise covariance matrix of the factor analysis variational distribution.
        max_grad_norm: Maximum norm for gradients which are used to update the parameters of the variational
            distribution.
        batch_size: The batch size to use for mini-batch gradient optimisation.
        n_epochs: The number of epochs for which to run the mini-batch gradient optimisation.
        results_output_dir: The path to directory where experiment results (including plots) will be saved.

    Returns:
        A dictionary with the following keys:
            - relative_distance_from_mean: The Frobenius norm between the mean of the true posterior and the mean of the
                variational posterior (including bias), divided by the Frobenius norm of the mean of the true
                posterior.
            - relative_distance_from_covar: The Frobenius norm between the covariance of the true posterior and the
                covariance of the variational posterior (not including bias), divided by the Frobenius norm of the
                covariance of the true posterior.
            - wasserstein_distance: The 2-Wasserstein distance between the true posterior and the variational posterior
                (not including bias).
            - alpha: The precision of the prior.
            - beta: The precision of the label noise.
            - dataset: The name of the dataset.
    """
    X, y = get_features_and_targets(dataset)
    n_samples, n_features = X.shape

    true_mean, true_covar, true_bias, alpha, beta = get_true_posterior(X, y)

    model = FeedForwardGaussianNet(
        input_dim=n_features,
        bias=True,
        loss_multiplier=n_samples,
        target_variance=1 / beta,
        random_seed=1,
    )

    variational_callback = FactorAnalysisVariationalInferenceCallback(
        latent_dim=latent_dim,
        precision=alpha,
        n_gradients_per_update=n_gradients_per_update,
        optimiser_class=optimiser_class,
        bias_optimiser_kwargs=bias_optimiser_kwargs,
        factors_optimiser_kwargs=factors_optimiser_kwargs,
        noise_optimiser_kwargs=noise_optimiser_kwargs,
        max_grad_norm=max_grad_norm,
        random_seed=1,
    )

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    trainer = Trainer(max_epochs=n_epochs, callbacks=variational_callback, progress_bar_refresh_rate=0)
    trainer.fit(model, train_dataloader=dataloader)

    variational_mean, variational_covar, variational_bias = get_variational_posterior(variational_callback)

    true_diag_covar, true_non_diag_covar = split_covariance(true_covar.numpy())
    variational_diag_covar, variational_non_diag_covar = split_covariance(variational_covar.numpy())

    generate_and_save_mean_plot(true_mean.numpy(), variational_mean.numpy(), results_output_dir, dataset_label)

    generate_and_save_variance_plot(true_diag_covar, variational_diag_covar, results_output_dir, dataset_label)

    generate_and_save_covariance_plot(
        true_non_diag_covar, variational_non_diag_covar, results_output_dir, dataset_label,
    )

    results = compute_metrics(true_mean, true_covar, true_bias, variational_mean, variational_covar, variational_bias)
    results['alpha'] = alpha
    results['beta'] = beta
    results['dataset'] = dataset_label

    return results


def get_true_posterior(X: Tensor, y: Tensor) -> (Tensor, Tensor, float, float, float):
    """
    Get the parameters of the true posterior of a linear regression model fit to the given data.

    Args:
        X: The features, of shape (n_samples, n_features).
        y: The targets, of shape (n_samples,).

    Returns:
        mean: The posterior mean, of shape (n_features,).
        covar: The posterior covariance, of shape (n_features, n_features).
        bias: The posterior bias.
        alpha: The precision of the Gaussian prior.
        beta: The precision of Gaussian target noise.
    """
    br = BayesianRidge()
    br.fit(X.numpy(), y.numpy())
    mean = torch.from_numpy(br.coef_).float()
    covar = torch.from_numpy(br.sigma_).float()
    bias = br.intercept_
    alpha = br.lambda_
    beta = br.alpha_

    return mean, covar, bias, alpha, beta


def get_variational_posterior(variational_callback: FactorAnalysisVariationalInferenceCallback,
                              ) -> (Tensor, Tensor, float):
    """
    Get the parameters of the linear regression posterior estimated by the given variational inference callback.

    Args:
        variational_callback: Variational inference callback. It is assumed that the bias term of the posterior
            corresponds to the final dimension of the mean and covariance.

    Returns:
        mean: The posterior mean, of shape (n_features,).
        covar: The posterior covariance, of shape (n_features, n_features).
        bias: The posterior bias.
    """
    weights = variational_callback.get_variational_mean()
    mean = weights[:-1]
    bias = weights[-1].item()
    covar = variational_callback.get_variational_covariance()[:-1, :-1]

    return mean, covar, bias


def split_covariance(covar: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Split the given covariance matrix into diagonal and non-diagonal entries.

    Args:
        covar: Covariance matrix of shape (n_features, n_features).

    Returns:
        diag_covar: Diagonal covariance entries of shape (n_features,).
        non_diag_covar: Non-diagonal covariance entries of shape (n_features, n_features). Diagonal is set to 0.
    """
    diag_covar = np.diag(covar)

    non_diag_covar = covar.copy()
    np.fill_diagonal(non_diag_covar, 0)

    return diag_covar, non_diag_covar


def compute_metrics(true_mean: Tensor, true_covar: Tensor, true_bias: float, variational_mean: Tensor,
                    variational_covar: Tensor, variational_bias: float) -> Dict[str, float]:
    """

    Args:
        true_mean: The true posterior mean, of shape (n_features,).
        true_covar: The true posterior covariance, of shape (n_features, n_features).
        true_bias: The true posterior bias.
        variational_mean: The variational posterior mean, of shape (n_features,).
        variational_covar: The variational posterior covariance, of shape (n_features, n_features).
        variational_bias: The variational posterior bias.

    Returns:
        A dictionary with the following keys:
            - relative_distance_from_mean: The Frobenius norm between the mean of the true posterior and the mean of the
                variational posterior (including bias), divided by the Frobenius norm of the mean of the true
                posterior.
            - relative_distance_from_covar: The Frobenius norm between the covariance of the true posterior and the
                covariance of the variational posterior (not including bias), divided by the Frobenius norm of the
                covariance of the true posterior.
            - wasserstein_distance: The 2-Wasserstein distance between the true posterior and the variational posterior
                (not including bias).
    """
    true_weights = torch.cat([true_mean, torch.Tensor([true_bias])])
    variational_weights = torch.cat([variational_mean, torch.Tensor([variational_bias])])

    distance_between_weights = compute_distance_between_matrices(true_weights, variational_weights)
    distance_between_covar = compute_distance_between_matrices(true_covar, variational_covar)

    true_weights_norm = compute_distance_between_matrices(true_weights, torch.zeros_like(true_weights))
    true_covar_norm = compute_distance_between_matrices(true_covar, torch.zeros_like(true_covar))

    return dict(
        relative_distance_from_mean=distance_between_weights / true_weights_norm,
        relative_distance_from_covar=distance_between_covar / true_covar_norm,
        wasserstein_distance=compute_gaussian_wasserstein_distance(
            true_mean, true_covar, variational_mean, variational_covar,
        )
    )


def generate_and_save_mean_plot(true_mean: np.ndarray, variational_mean: np.ndarray, plot_dir: str, dataset_label: str):
    """
    Generate and save a bar plot which compares the true and variational posterior means.

    Plot will be saved to '{plot_dir}/{dataset_label}_posterior_mean.png'

    Args:
        true_mean: The true posterior mean, of shape (n_features,).
        variational_mean: The variational posterior mean, of shape (n_features,).
        plot_dir: The directory for saving the plot.
        dataset_label: A label for the dataset which the posteriors correspond to.
    """
    plt.rcParams.update({'font.size': 15})

    plot_data = pd.DataFrame({
        'True': true_mean,
        'VIFA': variational_mean,
    }, index=range(1, len(true_mean) + 1))

    plot_data.plot(kind='bar', figsize=(16, 6))

    plt.xlabel('Feature index')
    plt.ylabel('Weight mean')
    plt.xticks(rotation=0)

    png_path = os.path.join(plot_dir, f'{dataset_label}_posterior_mean.png')
    plt.savefig(png_path, format='png')
    plt.close()


def generate_and_save_variance_plot(true_var: np.ndarray, variational_var: np.ndarray, plot_dir: str,
                                    dataset_label: str):
    """
    Generate and save a bar plot which compares the true and variational posterior variances.

    Plot will be saved to '{plot_dir}/{dataset_label}_posterior_variance.png'

    Args:
        true_var: The true posterior variances, of shape (n_features,).
        variational_var: The variational posterior variances, of shape (n_features,).
        plot_dir: The directory for saving the plot.
        dataset_label: A label for the dataset which the posteriors correspond to.
    """
    plt.rcParams.update({'font.size': 15})

    plot_data = pd.DataFrame({
        'True': true_var,
        'VIFA': variational_var,
    }, index=range(1, len(true_var) + 1))

    plot_data.plot(kind='bar', figsize=(16, 6))

    plt.xlabel('Feature index')
    plt.ylabel('Weight variance')
    plt.xticks(rotation=0)

    png_path = os.path.join(plot_dir, f'{dataset_label}_posterior_variance.png')
    plt.savefig(png_path, format='png')
    plt.close()


def generate_and_save_covariance_plot(true_covar: np.ndarray, variational_covar: np.ndarray, plot_dir: str,
                                      dataset_label: str):
    """
    Generate and save a bar plot which compares the true and variational posterior covariances.

    Plot will be saved to '{plot_dir}/{dataset_label}_posterior_covariance.png'

    Args:
        true_covar: The true posterior covariance, of shape (n_features, n_features).
        variational_covar: The variational posterior covariance, of shape (n_features, n_features).
        plot_dir: The directory for saving the plot.
        dataset_label: A label for the dataset which the posteriors correspond to.
    """
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    cmap = plt.cm.seismic
    true_img = ax[0].imshow(true_covar, cmap=cmap)
    variational_img = ax[1].imshow(variational_covar, cmap=cmap)

    ticks = np.arange(len(true_covar))
    tick_labels = ticks + 1
    for a in ax:
        a.set_xticks(ticks)
        a.set_xticklabels(tick_labels)

        a.set_yticks(ticks)
        a.set_yticklabels(tick_labels)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(variational_img, cax=cbar_ax)

    png_path = os.path.join(plot_dir, f'{dataset_label}_posterior_covariance.png')
    plt.savefig(png_path, format='png')
    plt.close()


@click.command()
@click.option('--boston-housing-input-path', type=str, help='The parquet file path to load the Boston Housing dataset')
@click.option('--yacht-hydrodynamics-input-path', type=str, help='The parquet file path to load the Yacht '
                                                                 'Hydrodynamics dataset')
@click.option('--concrete-strength-input-path', type=str, help='The parquet file path to load the Concrete '
                                                               'Compressive Strength dataset')
@click.option('--energy-efficiency-input-path', type=str, help='The parquet file path to load the Energy Efficiency '
                                                               'dataset')
@click.option('--results-output-dir', type=str, help='The directory path to save the results of the experiments')
def main(boston_housing_input_path: str, yacht_hydrodynamics_input_path: str, concrete_strength_input_path: str,
         energy_efficiency_input_path: str, results_output_dir: str):
    """
    Run experiments to estimate the posterior distribution of the weights of linear regression models via variational
    inference.

    Args:
        boston_housing_input_path: The parquet file path to load the Boston Housing dataset.
        yacht_hydrodynamics_input_path: The parquet file path to load the Yacht Hydrodynamics dataset.
        concrete_strength_input_path: The parquet file path to load the Concrete Compressive Strength dataset.
        energy_efficiency_input_path: The parquet file path to load the Energy Efficiency dataset.
        results_output_dir: The directory path to save the results of the experiments.
    """
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)['linear_regression_vi']

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

    Path(results_output_dir).mkdir(parents=True, exist_ok=True)

    results = run_all_experiments(
        datasets=datasets,
        dataset_labels=dataset_labels,
        dataset_params=params['dataset_params'],
        testing=params['testing'],
        results_output_dir=results_output_dir,
    )

    results.to_csv(os.path.join(results_output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
