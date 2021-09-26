import os
from pathlib import Path
from typing import List

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch.nn.functional import mse_loss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
import numpy as np
import pandas as pd
import click
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from swafa.models import FeedForwardNet
from swafa.callbacks import WeightPosteriorCallback
from swafa.fa import OnlineGradientFactorAnalysis, OnlineEMFactorAnalysis, OnlineFactorAnalysis
from swafa.posterior import ModelPosterior
from experiments.utils.factory import OPTIMISER_FACTORY


def run_all_experiments(
        datasets: List[pd.DataFrame],
        dataset_labels: List[str],
        latent_dim: int,
        n_folds: int,
        lr_pretrain: float,
        lr_swa: float,
        n_epochs_pretrain: int,
        n_epochs_swa: int,
        n_batches_per_epoch: int,
        weight_decay: float,
        gradient_optimiser: str,
        gradient_optimiser_kwargs: dict,
        gradient_warm_up_time_steps: int,
        em_warm_up_time_steps: int,
        n_posterior_samples: int,
) -> pd.DataFrame:
    """
    Run experiments on the given datasets.

    For each dataset, train a linear model to predict the target variable via SGD. Split training up into two parts, a
    pre-training phase followed by a second phase during which the weight vectors sampled after each batch update are
    used to estimate the posterior distribution of the weights via online gradient factor analysis (FA) and online
    expectation-maximisation (EM) FA.

    After training, sample weight vectors from the posteriors and use them to construct ensembles. Compute the test mean
    squared error (MSE) of the pre-trained weights, the average weights (SWA solution) and the two ensembles constructed
    from the online gradient FA posterior and the online EM FA posterior.

    Args:
        datasets: A list of datasets. Each dataset contains features and a target variable, where the target variable is
            in the final column.
        dataset_labels: A label for each of the datasets.
        latent_dim: The latent dimension of the FA models.
        n_folds: The number of folds of cross-validation to run for each dataset.
        lr_pretrain: The learning rate to use during the pre-training phase.
        lr_swa: The learning rate to use while sampling weight vectors after pre-training.
        n_epochs_pretrain: The number of pre-training epochs.
        n_epochs_swa: The number of epochs for which to sample weight vector after pre-training.
        n_batches_per_epoch: The number of batches per training epoch.
        weight_decay: The L2 regularisation strength to use while training.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient FA learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running means of the FA model in the
            online EM algorithm before updating the other parameters.
        n_posterior_samples: The number of samples of the weight vector to draw from each posterior to form each
            ensemble.

    Returns:
        The results of each cross-validation fold. The number of rows in the DataFrame is equal to n_datasets * n_folds.
        The DataFrame has the following columns:
            - mse_pretrained: (float) The test MSE of the pre-trained weight vector.
            - mse_swa: (float) The test MSE of the average weight vector (SWA solution).
            - mse_gradient_fa: (float) The test MSE of the ensemble constructed from the online gradient FA posterior.
            - mse_em_fa: (float) The test MSE of the ensemble constructed from the online EM FA posterior.
            - dataset: (str) The name of the dataset.
            - fold: (int) The index of the cross-validation fold.
    """
    results = []
    for label, dataset in zip(dataset_labels, datasets):
        print(f'Running experiments on {label} dataset...')
        print('-' * 100)

        dataset_results = run_dataset_experiments(
            dataset=dataset,
            dataset_label=label,
            latent_dim=latent_dim,
            n_folds=n_folds,
            lr_pretrain=lr_pretrain,
            lr_swa=lr_swa,
            n_epochs_pretrain=n_epochs_pretrain,
            n_epochs_swa=n_epochs_swa,
            n_batches_per_epoch=n_batches_per_epoch,
            weight_decay=weight_decay,
            gradient_optimiser=gradient_optimiser,
            gradient_optimiser_kwargs=gradient_optimiser_kwargs,
            gradient_warm_up_time_steps=gradient_warm_up_time_steps,
            em_warm_up_time_steps=em_warm_up_time_steps,
            n_posterior_samples=n_posterior_samples,
        )

        results.append(dataset_results)
        print('-' * 100)

    return pd.concat(results, ignore_index=True)


def run_dataset_experiments(
        dataset: pd.DataFrame,
        dataset_label: str,
        latent_dim: int,
        n_folds: int,
        lr_pretrain: float,
        lr_swa: float,
        n_epochs_pretrain: int,
        n_epochs_swa: int,
        n_batches_per_epoch: int,
        weight_decay: float,
        gradient_optimiser: str,
        gradient_optimiser_kwargs: dict,
        gradient_warm_up_time_steps: int,
        em_warm_up_time_steps: int,
        n_posterior_samples: int,
) -> pd.DataFrame:
    """
    Run experiments on the given dataset.

    Train a linear model to predict the target variable via SGD. Split training up into two parts, a pre-training phase
    followed by a second phase during which the weight vectors sampled after each batch update are used to estimate the
    posterior distribution of the weights via online gradient factor analysis (FA) and online expectation-maximisation
    (EM) FA.

    After training, sample weight vectors from the posteriors and use them to construct ensembles. Compute the test mean
    squared error (MSE) of the pre-trained weights, the average weights (SWA solution) and the two ensembles constructed
    from the online gradient FA posterior and the online EM FA posterior.

    Args:
        dataset: Contains features and a target variable, where the target variable is in the final column.
        dataset_label: A label for the dataset.
        latent_dim: The latent dimension of the FA models.
        n_folds: The number of folds of cross-validation to run for each dataset.
        lr_pretrain: The learning rate to use during the pre-training phase.
        lr_swa: The learning rate to use while sampling weight vectors after pre-training.
        n_epochs_pretrain: The number of pre-training epochs.
        n_epochs_swa: The number of epochs for which to sample weight vector after pre-training.
        n_batches_per_epoch: The number of batches per training epoch.
        weight_decay: The L2 regularisation strength to use while training.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient FA learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running means of the FA model in the
            online EM algorithm before updating the other parameters.
        n_posterior_samples: The number of samples of the weight vector to draw from each posterior to form each
            ensemble.

    Returns:
        The results of each cross-validation fold. The number of rows in the DataFrame is equal to n_folds. The
        DataFrame has the following columns:
            - mse_pretrained: (float) The test MSE of the pre-trained weight vector.
            - mse_swa: (float) The test MSE of the average weight vector (SWA solution).
            - mse_gradient_fa: (float) The test MSE of the ensemble constructed from the online gradient FA posterior.
            - mse_em_fa: (float) The test MSE of the ensemble constructed from the online EM FA posterior.
            - dataset: (str) The name of the dataset.
            - fold: (int) The index of the cross-validation fold.
    """
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    batch_size = int(np.floor(len(dataset) / n_batches_per_epoch))

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = []
    for k, (train_index, test_index) in enumerate(kfold.split(X)):
        print(f'Running cross-validation fold {k + 1} of {n_folds}...')

        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        fold_results = run_cv_fold(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                latent_dim=latent_dim,
                lr_pretrain=lr_pretrain,
                lr_swa=lr_swa,
                n_epochs_pretrain=n_epochs_pretrain,
                n_epochs_swa=n_epochs_swa,
                batch_size=batch_size,
                weight_decay=weight_decay,
                gradient_optimiser=gradient_optimiser,
                gradient_optimiser_kwargs=gradient_optimiser_kwargs,
                gradient_warm_up_time_steps=gradient_warm_up_time_steps,
                em_warm_up_time_steps=em_warm_up_time_steps,
                n_posterior_samples=n_posterior_samples,
                model_random_seed=k,
                posterior_random_seed=k + 1,
        )

        fold_results['dataset'] = dataset_label
        fold_results['fold'] = k + 1

        results.append(fold_results)

        print('-' * 100)

    return pd.DataFrame(results)


def run_cv_fold(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        latent_dim: int,
        lr_pretrain: float,
        lr_swa: float,
        n_epochs_pretrain: int,
        n_epochs_swa: int,
        batch_size: int,
        weight_decay: float,
        gradient_optimiser: str,
        gradient_optimiser_kwargs: dict,
        gradient_warm_up_time_steps: int,
        em_warm_up_time_steps: int,
        n_posterior_samples: int,
        model_random_seed: int,
        posterior_random_seed: int,
) -> dict:
    """
    Run a single cross-validation fold for the given data.

    Train a linear model to predict the target variable via SGD. Split training up into two parts, a pre-training phase
    followed by a second phase during which the weight vectors sampled after each batch update are used to estimate the
    posterior distribution of the weights via online gradient factor analysis (FA) and online expectation-maximisation
    (EM) FA.

    After training, sample weight vectors from the posteriors and use them to construct ensembles. Compute the test mean
    squared error (MSE) of the pre-trained weights, the average weights (SWA solution) and the two ensembles constructed
    from the online gradient FA posterior and the online EM FA posterior.

    Args:
        X_train: The training features, of shape (n_train, n_features).
        y_train: The training targets, of shape (n_train,).
        X_test: The test features, of shape (n_test, n_features).
        y_test: The test targets, of shape (n_test,).
        latent_dim: The latent dimension of the FA models.
        lr_pretrain: The learning rate to use during the pre-training phase.
        lr_swa: The learning rate to use while sampling weight vectors after pre-training.
        n_epochs_pretrain: The number of pre-training epochs.
        n_epochs_swa: The number of epochs for which to sample weight vector after pre-training.
        batch_size: The number of data points per training batch.
        weight_decay: The L2 regularisation strength to use while training.
        gradient_optimiser: The name of the PyTorch optimiser used in the online gradient FA learning algorithm. Options
            are 'sgd' and 'adam'.
        gradient_optimiser_kwargs: Keyword arguments for the PyTorch optimiser used in the online gradient FA learning
            algorithm.
        gradient_warm_up_time_steps: The number of time steps on which to update the running mean of the FA model in the
            online gradient algorithm before updating the other parameters.
        em_warm_up_time_steps: The number of time steps on which to update the running means of the FA model in the
            online EM algorithm before updating the other parameters.
        n_posterior_samples: The number of samples of the weight vector to draw from each posterior to form each
            ensemble.
        model_random_seed: The random seed to use when initialising the model.
        posterior_random_seed: The random seed to use when initialising the FA posteriors.

    Returns:
        The results of the cross-validation fold. Has the following keys:
            - mse_pretrained: (float) The test MSE of the pre-trained weight vector.
            - mse_swa: (float) The test MSE of the average weight vector (SWA solution).
            - mse_gradient_fa: (float) The test MSE of the ensemble constructed from the online gradient FA posterior.
            - mse_em_fa: (float) The test MSE of the ensemble constructed from the online EM FA posterior.
    """
    scaler = StandardScaler()
    X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
    X_test = torch.from_numpy(scaler.transform(X_test)).float()

    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    model_optimiser_kwargs = dict(lr=lr_pretrain, weight_decay=weight_decay)

    gradient_weight_posterior_kwargs = dict(
        latent_dim=latent_dim,
        optimiser=OPTIMISER_FACTORY[gradient_optimiser],
        optimiser_kwargs=gradient_optimiser_kwargs,
        n_warm_up_time_steps=gradient_warm_up_time_steps,
        random_seed=posterior_random_seed,
    )

    em_weight_posterior_kwargs = dict(
        latent_dim=latent_dim,
        n_warm_up_time_steps=em_warm_up_time_steps,
        random_seed=posterior_random_seed,
    )

    model, gradient_posterior_update_callback, em_posterior_update_callback = build_model_and_callbacks(
        X=X_train,
        model_optimiser_kwargs=model_optimiser_kwargs,
        gradient_weight_posterior_kwargs=gradient_weight_posterior_kwargs,
        em_weight_posterior_kwargs=em_weight_posterior_kwargs,
        model_random_seed=model_random_seed,
    )

    callbacks = [gradient_posterior_update_callback, em_posterior_update_callback]

    w_pretrained, b_pretrained = fit_model(
        X=X_train,
        y=y_train,
        model=model,
        callbacks=callbacks,
        n_epochs_pretrain=n_epochs_pretrain,
        n_epochs_swa=n_epochs_swa,
        lr_swa=lr_swa,
        batch_size=batch_size,
    )

    mse_pretrained, mse_swa, mse_gradient_fa, mse_em_fa = evaluate_model(
        X=X_test,
        y=y_test,
        w_pretrained=w_pretrained,
        b_pretrained=b_pretrained,
        gradient_posterior=gradient_posterior_update_callback.posterior,
        em_posterior=em_posterior_update_callback.posterior,
        n_posterior_samples=n_posterior_samples,
        random_seed=posterior_random_seed,
    )

    return dict(
        mse_pretrained=mse_pretrained,
        mse_swa=mse_swa,
        mse_gradient_fa=mse_gradient_fa,
        mse_em_fa=mse_em_fa,
    )


def build_model_and_callbacks(
        X: Tensor,
        model_optimiser_kwargs: dict,
        gradient_weight_posterior_kwargs: dict,
        em_weight_posterior_kwargs: dict,
        model_random_seed: int,
) -> (FeedForwardNet, WeightPosteriorCallback, WeightPosteriorCallback):
    """
    Build a linear model and callbacks which should be called during training to update the weight posteriors.

    Args:
        X: The features. Of shape (n_samples, n_features).
        model_optimiser_kwargs: Keyword arguments for the SGD optimiser used to train the linear model during the
            pre-training phase.
        gradient_weight_posterior_kwargs: Keyword arguments for the instance of OnlineGradientFactorAnalysis used to
            estimate the posterior.
        em_weight_posterior_kwargs: Keyword arguments for the instance of OnlineEMFactorAnalysis used to estimate the
            posterior.
        model_random_seed: The random seed used to initialise the linear model.

    Returns:
        model: An linear model with the same dimension as the input data plus a bias term.
        gradient_posterior_update_callback: Callbacks used to update the OnlineGradientFactorAnalysis weight posterior.
        em_posterior_update_callback: Callbacks used to update the OnlineEMFactorAnalysis weight posterior.
    """
    model = FeedForwardNet(
        input_dim=X.shape[1],
        bias=True,
        optimiser_class=SGD,
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
        update_epoch_start=1,
    )

    em_posterior_update_callback = WeightPosteriorCallback(
        posterior=em_posterior.weight_posterior,
        update_epoch_start=1,
    )

    return model, gradient_posterior_update_callback, em_posterior_update_callback


def fit_model(
        X: Tensor,
        y: Tensor,
        model: FeedForwardNet,
        callbacks: List[Callback],
        n_epochs_pretrain: int,
        n_epochs_swa: int,
        lr_swa: float,
        batch_size: int,
) -> (Tensor, Tensor):
    """
    Fit the given model to the given data.

    Training is split into two parts, a pre-training phase followed by a second phase during which the weight vectors
    sampled after each batch update are used to estimate the posterior distribution of the weights via online gradient
    factor analysis (FA) and online expectation-maximisation (EM) FA.

    Args:
        X: The features. Of shape (n_samples, n_features).
        y: The targets. Of shape (n_samples,).
        model: The model which is to be fit to the data.
        callbacks: Any callbacks which should be called during training.
        n_epochs_pretrain: The number of pre-training epochs.
        n_epochs_swa: The number of epochs for which to sample weight vector after pre-training.
        lr_swa: The learning rate to use while sampling weight vectors after pre-training.
        batch_size: The number of data points per training batch.

    Returns:
        A copy of the model's weights after the pre-training phase. Of shape (n_features, 1).
        A copy of the model's bias after the pre-training phase. Of shape (1,).
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    pre_trainer = Trainer(max_epochs=n_epochs_pretrain, progress_bar_refresh_rate=0)
    pre_trainer.fit(model, train_dataloader=dataloader)
    w_pretrained = torch.clone(model.output_layer.weight.data).reshape(-1, 1)
    b_pretrained = torch.clone(model.output_layer.bias.data).squeeze()

    swa_trainer = Trainer(max_epochs=n_epochs_swa, callbacks=callbacks, progress_bar_refresh_rate=0)
    model.optimiser_kwargs['lr'] = lr_swa
    swa_trainer.fit(model, train_dataloader=dataloader)

    return w_pretrained, b_pretrained


def evaluate_model(
        X: Tensor,
        y: Tensor,
        w_pretrained: Tensor,
        b_pretrained: Tensor,
        gradient_posterior: OnlineGradientFactorAnalysis,
        em_posterior: OnlineEMFactorAnalysis,
        n_posterior_samples: int,
        random_seed: int,
) -> (float, float, float, float):
    """
    Compute the mean squared error (MSE) of the pre-trained weights, the average weights (SWA solution) and the two
    ensembles constructed from the online gradient FA posterior and the online expectation-maximisation (EM) FA
    posterior.

    Args:
        X: The features. Of shape (n_samples, n_features).
        y: The targets. Of shape (n_samples,).
        w_pretrained: A copy of the model's weights after the pre-training phase. Of shape (n_features, 1).
        b_pretrained: A copy of the model's bias after the pre-training phase. Of shape (1,).
        gradient_posterior: The weight posterior estimated via online gradient FA.
        em_posterior: The weight posterior estimated via online EM FA.
        n_posterior_samples: The number of samples of the weight vector to draw from each posterior to form each
            ensemble.
        random_seed: The random seed to use when drawing samples from the posteriors.

    Returns:
        mse_pretrained: The MSE of the pre-trained weight vector.
        mse_swa: The MSE of the average weight vector (SWA solution).
        mse_gradient_fa: The MSE of the ensemble constructed from the online gradient FA posterior.
        mse_em_fa: The MSE of the ensemble constructed from the online EM FA posterior.
    """
    y_hat_pretrained = affine_transformation(X, w_pretrained, b_pretrained)

    y_hat_swa = swa_predict(X=X, posterior=gradient_posterior)

    y_hat_gradient_fa = posterior_ensemble_predict(
        X=X,
        posterior=gradient_posterior,
        n_posterior_samples=n_posterior_samples,
        random_seed=random_seed,
    )

    y_hat_em_fa = posterior_ensemble_predict(
        X=X,
        posterior=em_posterior,
        n_posterior_samples=n_posterior_samples,
        random_seed=random_seed,
    )

    mse_pretrained = mse_loss(y_hat_pretrained, y).item()
    mse_swa = mse_loss(y_hat_swa, y).item()
    mse_gradient_fa = mse_loss(y_hat_gradient_fa, y).item()
    mse_em_fa = mse_loss(y_hat_em_fa, y).item()

    return mse_pretrained, mse_swa, mse_gradient_fa, mse_em_fa


def swa_predict(X: Tensor, posterior: OnlineFactorAnalysis) -> Tensor:
    """
    Predict the targets of the given data using the SWA solution for the weight vector of a linear model.

    Args:
        X: The features. Of shape (n_samples, n_features).
        posterior: The weight posterior of the linear model. Note that posterior.get_mean() should return the average
            weights plus the bias term, with the bias term being the final element in the vector.

    Returns:
        The predicted targets. Of shape (n_samples,).
    """
    theta_swa = posterior.get_mean()
    w_swa = theta_swa[:-1].reshape(-1, 1)
    b_swa = theta_swa[-1]

    return affine_transformation(X, w_swa, b_swa)


def posterior_ensemble_predict(
        X: Tensor,
        posterior: OnlineFactorAnalysis,
        n_posterior_samples: int,
        random_seed: int,
) -> Tensor:
    """
    Predict the targets of the given data by sampling weight vectors from the posterior and using them to build an
    ensemble.

    Args:
        X: The features. Of shape (n_examples, n_features).
        posterior: The weight posterior of the linear model. Note that posterior.sample() should return weight vectors
            including the bias term, with the bias term being the final element in each vector.
        n_posterior_samples: The number of samples of the weight vector to draw from the posterior to form the ensemble.
        random_seed: The random seed to use when drawing samples from the posterior.

    Returns:
        The predicted targets. Of shape (n_examples,).
    """
    theta = posterior.sample(n_samples=n_posterior_samples, random_seed=random_seed).t()
    w = theta[:-1]
    b = theta[[-1]]

    return affine_transformation(X, w, b).mean(dim=1)


def affine_transformation(X: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """
    Compute an affine transformation.

    Args:
        X: Data of shape (n_samples, n_features).
        w: Weights of shape (n_features, n_models).
        b: Bias terms of shape (1, n_models).

    Returns:
        Outputs of shape (n_samples, n_models). If n_models = 1, squeeze to (n_samples,).
    """
    return (X.mm(w) + b).squeeze(dim=1)


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
    Run experiments to test whether linear regression predictions can be improved by stochastic weight averaging.

    Args:
        boston_housing_input_path: The parquet file path to load the Boston Housing dataset.
        yacht_hydrodynamics_input_path: The parquet file path to load the Yacht Hydrodynamics dataset.
        concrete_strength_input_path: The parquet file path to load the Concrete Compressive Strength dataset.
        energy_efficiency_input_path: The parquet file path to load the Energy Efficiency dataset.
        results_output_path: The parquet file path to save the experiment results.
    """
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)['linear_regression_predictions']

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
        latent_dim=params['latent_dim'],
        n_folds=params['n_folds'],
        lr_pretrain=params['lr_pretrain'],
        lr_swa=params['lr_swa'],
        n_epochs_pretrain=params['n_epochs_pretrain'],
        n_epochs_swa=params['n_epochs_swa'],
        n_batches_per_epoch=params['n_batches_per_epoch'],
        weight_decay=params['weight_decay'],
        gradient_optimiser=params['gradient_optimiser'],
        gradient_optimiser_kwargs=params['gradient_optimiser_kwargs'],
        gradient_warm_up_time_steps=params['gradient_warm_up_time_steps'],
        em_warm_up_time_steps=params['em_warm_up_time_steps'],
        n_posterior_samples=params['n_posterior_samples'],
    )

    print('Results:\n')
    print(results)

    Path(os.path.dirname(results_output_path)).mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_output_path)


if __name__ == '__main__':
    main()
