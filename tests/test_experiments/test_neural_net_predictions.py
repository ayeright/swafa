from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import pytest
from scipy import stats
from sklearn.preprocessing import StandardScaler
import torch

from experiments.neural_net_predictions import (
    aggregate_results,
    Objective,
    run_experiment,
    run_trial,
    train_test_split,
)
from swafa.callbacks import FactorAnalysisVariationalInferenceCallback
from swafa.models import FeedForwardGaussianNet
from swafa.utils import get_weight_dimension


class TestObjective:

    def test_unnormalise_target(self):
        objective = _init_objective()

        y = torch.Tensor([1, 2, 3])
        y_mean = 2
        y_scale = 5

        expected_y = torch.Tensor([7, 12, 17])

        actual_y = objective.de_standardise_target(y, y_mean, y_scale)

        assert torch.isclose(actual_y, expected_y).all()

    @pytest.mark.parametrize(
        "n_rows, n_columns, y_mean, y_scale",
        [
            (10, 3, 3, 5),
            (20, 5, -2, 2),
        ]
    )
    def test_predict(self, n_rows, n_columns, y_mean, y_scale):
        objective = _init_objective(n_rows, n_columns)
        model, variational_callback = _init_model_and_callback(n_rows, n_columns)
        X = torch.randn(n_rows, n_columns)

        y1 = objective.predict(model, variational_callback, X, y_mean, y_scale)
        y2 = objective.predict(model, variational_callback, X, y_mean, y_scale)

        assert y1.shape == (n_rows,)
        assert y2.shape == (n_rows,)
        assert not torch.isclose(y1, y2).all()

    @pytest.mark.parametrize(
        "n_rows, n_columns, y_mean, y_scale",
        [
            (10, 3, 3, 5),
            (20, 5, -2, 2),
        ]
    )
    def test_compute_bayesian_model_average(self, n_rows, n_columns, y_mean, y_scale):
        objective = _init_objective(n_rows, n_columns)
        model, variational_callback = _init_model_and_callback(n_rows, n_columns)
        X = torch.randn(n_rows, n_columns)

        mu, var = objective.compute_bayesian_model_average(model, variational_callback, X, y_mean, y_scale)

        assert mu.shape == (n_rows,)
        assert var.shape == (n_rows,)
        assert (var > 0).all()

    def test_compute_metrics(self):
        objective = _init_objective(n_rows=3)

        y = torch.Tensor([1, 2, 3])
        mu = torch.Tensor([2, 4, 0])
        var = torch.Tensor([1, 2, 1])

        expected_ll = np.mean(
            [
                -np.log(np.sqrt(var_i)) - np.log(np.sqrt(2 * np.pi)) - 0.5 * (y_i - mu_i) ** 2 / var_i
                for y_i, mu_i, var_i in zip(y.numpy(), mu.numpy(), var.numpy())
            ]
        )
        expected_rmse = np.sqrt(14 / 3)

        actual_ll, actual_rmse = objective.compute_metrics(y, mu, var)

        assert np.isclose(actual_ll, expected_ll)
        assert np.isclose(actual_rmse, expected_rmse)

    @pytest.mark.parametrize(
        "n_rows, n_columns, y_mean, y_scale",
        [
            (10, 3, 3, 5),
            (20, 5, -2, 2),
        ]
    )
    def test_test_model(self, n_rows, n_columns, y_mean, y_scale):
        objective = _init_objective(n_rows, n_columns)
        model, variational_callback = _init_model_and_callback(n_rows, n_columns)
        X = torch.randn(n_rows, n_columns)
        y = torch.randn(n_rows)

        ll, rmse = objective.test_model(model, variational_callback, X, y, y_mean, y_scale)

        assert ll < np.inf
        assert rmse < np.inf

    def test_train_model(self):
        n_rows = 10
        n_columns = 3

        objective = _init_objective(n_rows, n_columns)
        X = torch.randn(n_rows, n_columns)
        y = torch.randn(n_rows)

        model, variational_callback = objective.train_model(
            X, y, learning_rate=1e-3, prior_precision=1e-1, noise_precision=1e-1,
        )

        assert variational_callback.c is not None
        assert variational_callback.F is not None
        assert variational_callback.diag_psi is not None
        assert get_weight_dimension(model) == variational_callback.c.shape[0]

    def test_standardise_noise_precision(self):
        objective = _init_objective()

        noise_precision = 0.1
        y_scale = 2

        assert np.isclose(objective.standardise_noise_precision(noise_precision, y_scale), 0.4)

    @pytest.mark.parametrize(
        "n_rows, n_features",
        [
            (10, 3),
            (20, 5),
        ]
    )
    def test_transform_features_and_targets(self, n_rows, n_features):
        objective = _init_objective(n_rows, n_features)
        dataset = pd.DataFrame(np.random.randn(n_rows, n_features + 1))
        scaler = StandardScaler()
        scaler.fit(dataset.values)

        X, y = objective.transform_features_and_targets(dataset, scaler)

        assert torch.isclose(X.mean(dim=0), torch.zeros(n_features), atol=1e-5).all()
        assert torch.isclose(X.std(dim=0), torch.ones(n_features), atol=1e-1).all()

        assert torch.isclose(y.mean(), torch.zeros(1), atol=1e-5).all()
        assert torch.isclose(y.std(), torch.ones(1), atol=1e-1).all()

    @pytest.mark.parametrize(
        "n_rows, n_features",
        [
            (10, 3),
            (20, 5),
        ]
    )
    def test_fit_transform_features_and_targets(self, n_rows, n_features):
        objective = _init_objective(n_rows, n_features)
        dataset = pd.DataFrame(np.random.randn(n_rows, n_features + 1))

        X, y, scaler = objective.fit_transform_features_and_targets(dataset)

        assert torch.isclose(X.mean(dim=0), torch.zeros(n_features), atol=1e-5).all()
        assert torch.isclose(X.std(dim=0), torch.ones(n_features), atol=1e-1).all()

        assert torch.isclose(y.mean(), torch.zeros(1), atol=1e-5).all()
        assert torch.isclose(y.std(), torch.ones(1), atol=1e-1).all()

        assert scaler.mean_ is not None

    def test_train_and_test(self):
        objective = _init_objective(n_rows=40)

        train_index = np.arange(30)
        test_index = np.arange(30, 40)

        ll, rmse = objective.train_and_test(
            train_index, test_index, learning_rate=1e-3, prior_precision=1e-1, noise_precision=1e-1,
        )

        assert ll < np.inf
        assert rmse < np.inf

    @pytest.mark.parametrize("n_trials", [1, 3])
    def test_objective_in_study(self, n_trials):
        objective = _init_objective(n_rows=40)
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1), direction='maximize')

        study.optimize(objective, n_trials=n_trials)

        assert len(study.trials) == n_trials


def test_aggregate_results():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    results = pd.DataFrame({'a': a, 'b': b})

    expected_aggregated_results = pd.DataFrame(
        {'mean': [a.mean(), b.mean()], 'se': [stats.sem(a), stats.sem(b)]},
        index=['a', 'b'],
    )

    actual_aggregated_results = aggregate_results(results)

    assert np.isclose(actual_aggregated_results.values, expected_aggregated_results.values).all()
    assert (actual_aggregated_results.columns == expected_aggregated_results.columns).all()
    assert (actual_aggregated_results.index == expected_aggregated_results.index).all()


def test_train_test_split():
    n_rows = 50
    train_fraction = 0.8

    dataset = pd.DataFrame(np.random.randn(n_rows, 3))

    train_index, test_index = train_test_split(dataset, train_fraction)

    assert len(train_index) == 40
    assert len(test_index) == 10
    assert len(np.intersect1d(train_index, test_index)) == 0


@pytest.mark.parametrize("test", [True, False])
def test_run_trial(test):
    dataset = pd.DataFrame(np.random.randn(50, 3))

    two_trial_results = [
        run_trial(
            dataset=dataset,
            train_index=np.arange(40),
            test_index=np.arange(40, 50),
            n_hyperparameter_trials=2,
            n_cv_folds=2,
            latent_dim=2,
            n_gradients_per_update=2,
            max_grad_norm=10,
            batch_size=5,
            n_epochs=2,
            learning_rate_range=[1e-3, 1e-2],
            prior_precision_range=[1e-3, 1e-2],
            noise_precision_range=[1e-3, 1e-2],
            n_bma_samples=5,
            hidden_dims=[4],
            hidden_activation_fn=torch.nn.ReLU(),
            model_random_seed=1,
            test=test,
        )
        for _ in range(2)
    ]

    for results in two_trial_results:
        if test:
            assert set(results.keys()) == {'val_ll', 'test_ll', 'test_rmse'}
        else:
            assert set(results.keys()) == {'val_ll'}

    for key, val in two_trial_results[0].items():
        assert np.isclose(val, two_trial_results[1][key])


@pytest.mark.parametrize("test", [True, False])
def test_run_experiment(test):
    dataset = pd.DataFrame(np.random.randn(50, 3))

    two_experiment_results = [
        run_experiment(
            dataset=dataset,
            n_train_test_splits=2,
            train_fraction=0.8,
            n_hyperparameter_trials=2,
            n_cv_folds=2,
            latent_dim=2,
            n_gradients_per_update=2,
            max_grad_norm=10,
            batch_size=5,
            n_epochs=2,
            learning_rate_range=[1e-3, 1e-2],
            prior_precision_range=[1e-3, 1e-2],
            noise_precision_range=[1e-3, 1e-2],
            n_bma_samples=5,
            hidden_dims=[4],
            hidden_activation_fn=torch.nn.ReLU(),
            data_split_random_seed=1,
            test=test,
        )
        for _ in range(2)
    ]

    for results in two_experiment_results:
        assert set(results.columns) == {'mean', 'se'}

        if test:
            assert set(results.index) == {'val_ll', 'test_ll', 'test_rmse'}
        else:
            assert set(results.index) == {'val_ll'}

    assert np.isclose(two_experiment_results[0].values, two_experiment_results[1].values).all()


def _init_objective(
    n_rows: int = 10,
    n_columns: int = 3,
    n_cv_folds: int = 2,
    latent_dim: int = 2,
    n_gradients_per_update: int = 2,
    max_grad_norm: float = 10,
    batch_size: int = 5,
    n_epochs: int = 2,
    learning_rate_range: List[float] = None,
    prior_precision_range: List[float] = None,
    noise_precision_range: List[float] = None,
    n_bma_samples: int = 2,
    hidden_dims: Optional[List[int]] = None,
    hidden_activation_fn: Optional[torch.nn.Module] = None,
    random_seed: Optional[int] = 1,
) -> Objective:
    learning_rate_range = learning_rate_range or [1e-3, 1e-2]
    prior_precision_range = prior_precision_range or [1e-3, 1e-2]
    noise_precision_range = noise_precision_range or [1e-3, 1e-2]
    hidden_dims = hidden_dims or [4]
    hidden_activation_fn = hidden_activation_fn or torch.nn.ReLU()

    dataset = pd.DataFrame(np.random.randn(n_rows, n_columns))

    return Objective(
        dataset=dataset,
        n_cv_folds=n_cv_folds,
        latent_dim=latent_dim,
        n_gradients_per_update=n_gradients_per_update,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate_range=learning_rate_range,
        prior_precision_range=prior_precision_range,
        noise_precision_range=noise_precision_range,
        n_bma_samples=n_bma_samples,
        hidden_dims=hidden_dims,
        hidden_activation_fn=hidden_activation_fn,
        random_seed=random_seed,
    )


def _init_model_and_callback(
    n_samples: int = 10,
    n_features: int = 3,
    latent_dim: int = 2,
    prior_precision: float = 0.1,
    noise_precision: float = 0.1,
    n_gradients_per_update: int = 2,
    hidden_dims: Optional[List[int]] = None,
    hidden_activation_fn: Optional[torch.nn.Module] = None,
    random_seed: Optional[int] = 1,
) -> (FeedForwardGaussianNet, FactorAnalysisVariationalInferenceCallback):
    hidden_dims = hidden_dims or [4]
    hidden_activation_fn = hidden_activation_fn or torch.nn.ReLU()

    model = FeedForwardGaussianNet(
        input_dim=n_features,
        hidden_dims=hidden_dims,
        hidden_activation_fn=hidden_activation_fn,
        loss_multiplier=n_samples,
        target_variance=1 / noise_precision,
        random_seed=random_seed,
    )

    variational_callback = FactorAnalysisVariationalInferenceCallback(
        latent_dim=latent_dim,
        precision=prior_precision,
        n_gradients_per_update=n_gradients_per_update,
        optimiser_class=torch.optim.SGD,
        random_seed=random_seed,
    )

    variational_callback.on_fit_start(trainer=None, pl_module=model)

    return model, variational_callback
