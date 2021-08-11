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
from experiments.utils.callbacks import PosteriorEvaluationCallback
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
) -> pd.DataFrame:
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
        )

        results.append(dataset_results)
        print('-' * 100)

    return pd.concat(results)


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
) -> pd.DataFrame:
    X, y = get_features_and_targets(dataset)
    true_posterior_mean, true_posterior_covar, alpha, beta = compute_true_posterior(X, y)
    observation_dim = X.shape[1] - 1

    results = []
    for latent_dim in range(1, observation_dim):
        print(f'Using a posterior with latent dimension equal to {latent_dim} (of {observation_dim - 1})...')
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
        gradient_posterior_eval_callback,
        em_posterior_eval_callback,
    ) = build_model_and_callbacks(
        X=X,
        true_posterior_mean=true_posterior_mean,
        true_posterior_covar=true_posterior_covar,
        model_optimiser_class=OPTIMISER_FACTORY[model_optimiser],
        model_optimiser_kwargs=model_optimiser_kwargs,
        gradient_weight_posterior_kwargs=gradient_weight_posterior_kwargs,
        em_weight_posterior_kwargs=em_weight_posterior_kwargs,
        posterior_update_epoch_start=posterior_update_epoch_start,
        posterior_eval_epoch_frequency=posterior_eval_epoch_frequency,
        random_seed=model_random_seed,
    )

    callbacks = [
        gradient_posterior_update_callback,
        em_posterior_update_callback,
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

    results = collate_callback_results(gradient_posterior_eval_callback, em_posterior_eval_callback)

    return results


def get_features_and_targets(dataset: pd.DataFrame) -> (Tensor, Tensor):
    X = dataset.iloc[:, :-1].values
    X = torch.from_numpy(StandardScaler().fit_transform(X)).float()
    X = torch.cat([X, torch.ones(len(X), 1)], dim=1)
    y = torch.from_numpy(dataset.iloc[:, -1].values).float()
    return X, y


def compute_true_posterior(X: Tensor, y: Tensor, alpha: Optional[float] = None, beta: Optional[float] = None,
                           ) -> (Tensor, Tensor, float, float):
    beta = beta or compute_beta(y)
    S, alpha = compute_true_posterior_covar(X, beta, alpha)
    m = compute_true_posterior_mean(X, y, beta, S)
    return m, S, alpha, beta


def compute_beta(y: Tensor) -> float:
    sigma = torch.std(y)
    return (1 / (sigma ** 2)).item()


def compute_true_posterior_covar(X: Tensor, beta: float, alpha: Optional[float] = None) -> (Tensor, float):
    B = beta * torch.einsum('ij,ik->jk', X, X)
    alpha = alpha or torch.diag(B).max().item()
    A = alpha * torch.eye(len(B)) + B
    S = torch.linalg.inv(A)
    return S, alpha


def compute_true_posterior_mean(X: Tensor, y: Tensor, beta: float, S: Tensor) -> Tensor:
    b = beta * (y.reshape(-1, 1) * X).sum(dim=0, keepdims=True).t()
    return S.mm(b).squeeze()


def build_model_and_callbacks(
        X: Tensor,
        true_posterior_mean: Tensor,
        true_posterior_covar: Tensor,
        model_optimiser_class: Optimizer,
        model_optimiser_kwargs: dict,
        gradient_weight_posterior_kwargs: dict,
        em_weight_posterior_kwargs: dict,
        posterior_update_epoch_start: int,
        posterior_eval_epoch_frequency: int,
        random_seed: int,
) -> (FeedForwardNet, WeightPosteriorCallback, WeightPosteriorCallback):
    model = FeedForwardNet(
        input_dim=X.shape[1],
        bias=False,
        optimiser_class=model_optimiser_class,
        optimiser_kwargs=model_optimiser_kwargs,
        random_seed=random_seed,
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

    gradient_posterior_eval_callback = PosteriorEvaluationCallback(
        posterior=gradient_posterior.weight_posterior,
        true_mean=true_posterior_mean,
        true_covar=true_posterior_covar,
        eval_epoch_frequency=posterior_eval_epoch_frequency,
    )

    em_posterior_eval_callback = PosteriorEvaluationCallback(
        posterior=em_posterior.weight_posterior,
        true_mean=true_posterior_mean,
        true_covar=true_posterior_covar,
        eval_epoch_frequency=posterior_eval_epoch_frequency,
    )

    return (
        model,
        gradient_posterior_update_callback,
        em_posterior_update_callback,
        gradient_posterior_eval_callback,
        em_posterior_eval_callback,
    )


def fit_model(X: Tensor, y: Tensor, model: FeedForwardNet, callbacks: List[Callback], n_epochs: int, batch_size: int):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    trainer = Trainer(max_epochs=n_epochs, callbacks=callbacks, progress_bar_refresh_rate=0)
    trainer.fit(model, train_dataloader=dataloader)


def collate_callback_results(gradient_posterior_eval_callback: PosteriorEvaluationCallback,
                             em_posterior_eval_callback: PosteriorEvaluationCallback) -> pd.DataFrame:
    results = []
    for i, (epoch_gradient, epoch_em) in enumerate(zip(gradient_posterior_eval_callback.eval_epochs,
                                                       em_posterior_eval_callback.eval_epochs)):
        if epoch_gradient != epoch_em:
            raise RuntimeError(f'The evaluation epochs of the two evaluation callbacks must be equal, not '
                               f'{epoch_gradient} and {epoch_em}')

        results.append(dict(
            epoch=epoch_gradient,
            mean_distance_online_gradient=gradient_posterior_eval_callback.relative_mean_distances[i],
            covar_distance_online_gradient=gradient_posterior_eval_callback.relative_covar_distances[i],
            wasserstein_online_gradient=gradient_posterior_eval_callback.wasserstein_distances[i],
            mean_distance_online_em=em_posterior_eval_callback.relative_mean_distances[i],
            covar_distance_online_em=em_posterior_eval_callback.relative_covar_distances[i],
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
    )

    print('Results:\n')
    print(results)

    Path(os.path.dirname(results_output_path)).mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_output_path)


if __name__ == '__main__':
    main()
