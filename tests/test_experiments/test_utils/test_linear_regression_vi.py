import numpy as np
import pandas as pd
import pytest
import torch

from swafa.callbacks import FactorAnalysisVariationalInferenceCallback
from experiments.linear_regression_vi import (
    compute_metrics,
    get_true_posterior,
    get_variational_posterior,
    run_all_experiments,
    split_covariance,
    train_test_split,
)
from experiments.utils.metrics import compute_gaussian_wasserstein_distance


def test_train_test_split():
    n_rows, n_columns = 20, 10
    dataset = pd.DataFrame(np.random.rand(n_rows, n_columns))

    train_dataset, test_dataset = train_test_split(dataset)

    expected_n_rows = int(n_rows / 2)

    assert train_dataset.shape == (expected_n_rows, n_columns)
    assert test_dataset.shape == (expected_n_rows, n_columns)
    assert len(np.intersect1d(train_dataset.index, test_dataset.index)) == 0


def test_get_true_posterior():
    n_rows, n_columns = 20, 3
    X = torch.randn(n_rows, n_columns)
    y = torch.randn(n_rows)

    mean, covar, bias, alpha, beta = get_true_posterior(X, y)

    assert mean.shape == (n_columns,)
    assert covar.shape == (n_columns, n_columns)
    assert (torch.diag(covar) >= 0).all()
    assert isinstance(bias, float)
    assert alpha > 0
    assert beta > 0


@pytest.mark.parametrize("n_features, latent_dim",
        [
            (5, 1),
            (5, 3),
            (10, 10),
        ]
    )
def test_get_variational_posterior(n_features, latent_dim):
    variational_callback = FactorAnalysisVariationalInferenceCallback(latent_dim, precision=1)
    variational_callback.weight_dim = n_features + 1  # including bias term
    variational_callback._init_variational_params()

    mean, covar, bias = get_variational_posterior(variational_callback)

    assert mean.shape == (n_features,)
    assert covar.shape == (n_features, n_features)
    assert (torch.diag(covar) >= 0).all()
    assert isinstance(bias, float)


def test_split_covariance():
    d = 10
    covar = np.random.rand(d, d)

    diag_covar, non_diag_covar = split_covariance(covar)

    assert diag_covar.shape == (d,)
    assert non_diag_covar.shape == (d, d)
    assert (diag_covar == np.diag(covar)).all()
    assert (np.diag(non_diag_covar) == 0).all()


def test_compute_metrics():
    true_mean = torch.Tensor([1, 2])
    true_covar = torch.Tensor([
        [1, -1],
        [2, 3],
    ])
    true_bias = 4

    variational_mean = torch.Tensor([0, 3])
    variational_covar = torch.Tensor([
        [2, -3],
        [3, 2],
    ])
    variational_bias = 6

    expected_relative_distance_from_mean = np.sqrt(6) / np.sqrt(21)
    expected_relative_distance_from_covar = np.sqrt(7) / np.sqrt(15)
    expected_wasserstein_distance = compute_gaussian_wasserstein_distance(
        true_mean, true_covar, variational_mean, variational_covar,
    )

    actual_metrics = compute_metrics(
        true_mean, true_covar, true_bias, variational_mean, variational_covar, variational_bias,
    )

    assert np.isclose(actual_metrics['relative_distance_from_mean'], expected_relative_distance_from_mean)
    assert np.isclose(actual_metrics['relative_distance_from_covar'], expected_relative_distance_from_covar)
    assert np.isclose(actual_metrics['wasserstein_distance'], expected_wasserstein_distance)


@pytest.mark.parametrize("n_datasets", [1, 2, 3])
def test_run_all_experiments(tmpdir, n_datasets):
    datasets = [pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])] * n_datasets
    dataset_labels = [f'dataset{i}' for i in range(n_datasets)]
    params = dict(
        latent_dim=2,
        n_gradients_per_update=1,
        optimiser='sgd',
        bias_optimiser_kwargs=dict(lr=1e-3),
        factors_optimiser_kwargs=dict(lr=1e-3),
        noise_optimiser_kwargs=dict(lr=1e-3),
        max_grad_norm=1,
        batch_size=2,
        n_epochs=5,
    )
    dataset_params = {label: params for label in dataset_labels}

    train_results_1 = run_all_experiments(
        datasets, dataset_labels, dataset_params, testing=False, results_output_dir=tmpdir,
    )

    train_results_2 = run_all_experiments(
        datasets, dataset_labels, dataset_params, testing=False, results_output_dir=tmpdir,
    )

    test_results_1 = run_all_experiments(
        datasets, dataset_labels, dataset_params, testing=True, results_output_dir=tmpdir,
    )

    test_results_2 = run_all_experiments(
        datasets, dataset_labels, dataset_params, testing=True, results_output_dir=tmpdir,
    )

    metric_columns = [
        'relative_distance_from_mean',
        'relative_distance_from_covar',
        'wasserstein_distance',
    ]

    other_columns = [
        'alpha',
        'beta',
        'dataset',
    ]

    expected_columns = set(metric_columns + other_columns)

    for results in [train_results_1, train_results_2, test_results_1, test_results_2]:
        assert set(results.columns) == expected_columns
        assert len(results) == n_datasets
        assert set(results['dataset']) == set(dataset_labels)

    assert np.isclose(train_results_1[metric_columns].values, train_results_2[metric_columns].values).all()
    assert np.isclose(test_results_1[metric_columns].values, test_results_2[metric_columns].values).all()
    assert not np.isclose(train_results_1[metric_columns].values, test_results_1[metric_columns].values).all()
