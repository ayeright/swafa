import pytest
import numpy as np
import pandas as pd

from experiments.linear_regression_predictions import run_all_experiments


@pytest.mark.parametrize(
        "n_datasets, n_folds, n_samples, n_features",
        [
            (1, 2, [100], [2]),
            (1, 3, [50], [3]),
            (2, 2, [100, 50], [2, 3]),
            (2, 3, [100, 50], [2, 3]),
        ]
    )
def test_all_experiments_results_rows_and_columns(n_datasets, n_folds, n_samples, n_features):
    datasets = [pd.DataFrame(np.random.randn(n_samples[i], n_features[i] + 1)) for i in range(n_datasets)]
    dataset_labels = [f"dataset_{i}" for i in range(n_datasets)]

    results = run_all_experiments(
        datasets=datasets,
        dataset_labels=dataset_labels,
        latent_dim=2,
        n_folds=n_folds,
        lr_pretrain=1e-3,
        lr_swa=1e-1,
        n_epochs_pretrain=10,
        n_epochs_swa=10,
        n_batches_per_epoch=10,
        weight_decay=1e-4,
        gradient_optimiser='sgd',
        gradient_optimiser_kwargs=dict(lr=0.01),
        gradient_warm_up_time_steps=10,
        em_warm_up_time_steps=10,
        n_posterior_samples=10,
    )

    expected_columns = [
        'mse_pretrained',
        'mse_swa',
        'mse_gradient_fa',
        'mse_em_fa',
        'dataset',
        'fold',
    ]

    actual_columns = list(results.columns)
    assert len(actual_columns) == len(expected_columns)
    assert len(np.intersect1d(actual_columns, expected_columns)) == len(actual_columns)

    expected_n_rows = n_datasets * n_folds
    assert len(results) == expected_n_rows

    for i in range(n_datasets):
        assert (results['dataset'] == dataset_labels[i]).sum() == n_folds
