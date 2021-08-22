import pytest
import numpy as np
import pandas as pd
import torch

from experiments.linear_models import (
    compute_true_posterior,
    compute_true_posterior_covar,
    compute_true_posterior_mean,
    get_features_and_targets,
    run_all_experiments,
)


@pytest.mark.parametrize(
        "n_datasets, n_trials, n_features, n_epochs, posterior_eval_epoch_frequency",
        [
            (1, 1, [2], 1, 1),
            (1, 2, [3], 2, 1),
            (1, 2, [3], 4, 2),
            (2, 1, [2, 3], 1, 1),
            (2, 2, [3, 2], 2, 1),
            (2, 2, [3, 3], 4, 2),
        ]
    )
def test_all_experiments_results_rows_and_columns(n_datasets, n_trials, n_features, n_epochs,
                                                  posterior_eval_epoch_frequency):
    n_samples = 100
    datasets = [pd.DataFrame(np.random.randn(n_samples, n_features[i] + 1)) for i in range(n_datasets)]
    dataset_labels = [f"dataset_{i}" for i in range(n_datasets)]

    results = run_all_experiments(
        datasets=datasets,
        dataset_labels=dataset_labels,
        n_trials=n_trials,
        model_optimiser='sgd',
        model_optimiser_kwargs=dict(lr=0.01),
        n_epochs=n_epochs,
        batch_size=32,
        gradient_optimiser='adam',
        gradient_optimiser_kwargs=dict(lr=0.01),
        gradient_warm_up_time_steps=1,
        em_warm_up_time_steps=1,
        posterior_update_epoch_start=1,
        posterior_eval_epoch_frequency=posterior_eval_epoch_frequency,
        precision_scaling_factor=0.1,
    )

    expected_columns = [
        'epoch',
        'mean_distance_sklearn',
        'covar_distance_sklearn',
        'wasserstein_sklearn',
        'mean_distance_online_gradient',
        'covar_distance_online_gradient',
        'wasserstein_online_gradient',
        'mean_distance_online_em',
        'covar_distance_online_em',
        'wasserstein_online_em',
        'latent_dim',
        'trial',
        'mean_norm',
        'covar_norm',
        'alpha',
        'beta',
        'dataset',
        'n_samples',
        'observation_dim',
    ]

    actual_columns = list(results.columns)
    assert len(actual_columns) == len(expected_columns)
    assert len(np.intersect1d(actual_columns, expected_columns)) == len(actual_columns)

    expected_n_rows = sum([(n_features[i] - 1) * n_trials for i in range(n_datasets)]) * n_epochs \
        / posterior_eval_epoch_frequency
    assert len(results) == expected_n_rows

    for i in range(n_datasets):
        assert (results['dataset'] == dataset_labels[i]).sum() == (n_features[i] - 1) * n_trials * n_epochs \
                    / posterior_eval_epoch_frequency


@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("n_features", [2, 3])
def test_get_features_and_targets(n_samples, n_features):
    dataset = pd.DataFrame(np.random.randn(n_samples, n_features + 1))
    features = dataset.iloc[:, :-1].values
    targets = dataset.iloc[:, -1].values

    means = features.mean(axis=0, keepdims=True)
    stds = features.std(axis=0, keepdims=True)

    X, y = get_features_and_targets(dataset)

    assert np.isclose((X.numpy() * stds) + means, features, atol=1e-4).all()
    assert np.isclose(y.numpy(), targets).all()


@pytest.mark.parametrize("n_samples", [10, 100])
@pytest.mark.parametrize("n_features", [3, 8])
@pytest.mark.parametrize("alpha", [None, 0.1])
@pytest.mark.parametrize("beta", [0.01, 0.1])
@pytest.mark.parametrize("alpha_scaling_factor", [0.01, 0.1])
def test_compute_true_posterior_covar(n_samples, n_features, alpha, beta, alpha_scaling_factor):
    X = torch.randn(n_samples, n_features)

    actual_S, actual_alpha = compute_true_posterior_covar(
        X, beta, alpha=alpha, alpha_scaling_factor=alpha_scaling_factor,
    )

    xxt = torch.zeros(n_features, n_features)
    for x in X:
        x = x.reshape(-1, 1)
        xxt += x.mm(x.t())
    B = beta * xxt

    if alpha is None:
        assert np.isclose(actual_alpha, alpha_scaling_factor * torch.diag(B).mean().item())
    else:
        assert actual_alpha == alpha

    A = actual_alpha * torch.eye(n_features) + B
    expected_S = torch.linalg.inv(A)

    assert torch.isclose(actual_S, expected_S).all()


@pytest.mark.parametrize("n_samples", [10, 100])
@pytest.mark.parametrize("n_features", [3, 8])
@pytest.mark.parametrize("alpha", [None, 0.1])
@pytest.mark.parametrize("beta", [0.01, 0.1])
@pytest.mark.parametrize("alpha_scaling_factor", [0.01, 0.1])
def test_compute_true_posterior_mean(n_samples, n_features, alpha, beta, alpha_scaling_factor):
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples)

    S, _ = compute_true_posterior_covar(X, beta, alpha=alpha, alpha_scaling_factor=alpha_scaling_factor)

    actual_m = compute_true_posterior_mean(X, y, beta, S)

    yx = torch.zeros(n_features)
    for i, x in enumerate(X):
        yx += y[i] * x
    b = beta * yx

    expected_m = S.mm(b.reshape(-1, 1)).squeeze()
    assert torch.isclose(actual_m, expected_m, atol=1e-4).all()


@pytest.mark.parametrize("n_samples", [10, 100])
@pytest.mark.parametrize("n_features", [3, 8])
@pytest.mark.parametrize("alpha", [None, 0.1])
@pytest.mark.parametrize("beta", [None, 0.1])
@pytest.mark.parametrize("alpha_scaling_factor", [0.01, 0.1])
def test_compute_true_posterior(n_samples, n_features, alpha, beta, alpha_scaling_factor):
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples)

    actual_m, actual_S, actual_alpha, actual_beta = compute_true_posterior(
        X, y, alpha, beta, alpha_scaling_factor=alpha_scaling_factor,
    )

    if beta is None:
        assert np.isclose(actual_beta, 1 / torch.var(y).item())
    else:
        assert actual_beta == beta

    expected_S, expected_alpha = compute_true_posterior_covar(
        X, actual_beta, alpha=alpha, alpha_scaling_factor=alpha_scaling_factor,
    )
    assert torch.isclose(actual_S, expected_S).all()

    expected_m = compute_true_posterior_mean(X, y, actual_beta, expected_S)
    assert torch.isclose(actual_m, expected_m).all()
