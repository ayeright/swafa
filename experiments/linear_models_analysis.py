import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import click


def run_analysis(results: pd.DataFrame, analysis_output_dir: str):
    """
    Aggregate the experiment results and generate plots showing how similar the estimated posterior distributions are
    to the true posterior distributions.

    For each dataset and latent dimension, group by n_epochs and compute the mean and standard error of the metrics in
    the experiment results. Save these statistics to csv files.

    Also, for each dataset generate three plots, one showing the distance between the true and estimated posterior mean,
    one showing the distance between the true and estimated posterior covariance matrix and one showing the Wasserstein
    distance between the true and estimated posteriors. Save these plots to png files.

    Args:
        results: The results of each experiment. The number of rows in the DataFrame is equal to
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
        analysis_output_dir: The directory path to save the output of the analysis.
    """
    metric_suffixes = ['sklearn', 'online_gradient', 'online_em']
    mean_columns = [f'mean_distance_{x}' for x in metric_suffixes]
    covar_columns = [f'covar_distance_{x}' for x in metric_suffixes]
    wasserstein_columns = [f'wasserstein_{x}' for x in metric_suffixes]

    results[mean_columns] = results[mean_columns].values / results[['mean_norm']].values
    results[covar_columns] = results[covar_columns].values / results[['covar_norm']].values

    axis_titles = ['Batch SVD', 'Online SGA', 'Online EM']

    for dataset_label in results['dataset'].unique():
        dataset_means, dataset_standard_errors = aggregate_experiment_results(
            results,
            dataset_label,
            metric_columns=mean_columns + covar_columns + wasserstein_columns,
        )

        dataset_means.to_csv(os.path.join(
            analysis_output_dir, f'linear_model_metric_means__{dataset_label}.csv'), index=False,
        )

        dataset_standard_errors.to_csv(os.path.join(
            analysis_output_dir, f'linear_model_metric_standard_errors__{dataset_label}.csv'), index=False,
        )

        generate_and_save_error_bar_plot(
            dataset_means,
            dataset_standard_errors,
            png_path=os.path.join(analysis_output_dir, f'linear_models_mean_distance__{dataset_label}.png'),
            ylabel='Relative mean distance',
            axes_columns=mean_columns,
            axes_titles=axis_titles,
        )

        generate_and_save_error_bar_plot(
            dataset_means,
            dataset_standard_errors,
            png_path=os.path.join(analysis_output_dir, f'linear_models_covariance_distance__{dataset_label}.png'),
            ylabel='Relative covariance distance',
            axes_columns=covar_columns,
            axes_titles=axis_titles,
        )

        generate_and_save_error_bar_plot(
            dataset_means,
            dataset_standard_errors,
            png_path=os.path.join(analysis_output_dir, f'linear_models_wasserstein__{dataset_label}.png'),
            ylabel='2-Wasserstein distance',
            axes_columns=wasserstein_columns,
            axes_titles=axis_titles,
        )


def aggregate_experiment_results(results: pd.DataFrame, dataset_label: str, metric_columns: List[str],
                                 ) -> (pd.DataFrame, pd.DataFrame):
    """
    For the given dataset, group the results by latent dimension and epoch and average the given metric columns over all
    trials.

    Also, compute the standard error of the mean of each group.

    Args:
        results: The results of each experiment. Columns must contain 'dataset', 'latent_dim', 'epoch' and
            metric_columns.
        dataset_label: The name of the dataset for which to aggregate the results.
        metric_columns: The columns which are to be aggregated.

    Returns:
        group_means: The mean of each column in metric_columns for each combination of latent dimension and epoch. Has
            columns 'latent_dim', 'epoch' and metric_columns.
        group_standard_errors: The standard error of each value in group_means. Has the same shape as group_means.
    """
    dataset_results = results[results['dataset'] == dataset_label]
    grouped_results = dataset_results.groupby(['latent_dim', 'epoch'])
    group_means = grouped_results[metric_columns].mean().reset_index()
    group_standard_errors = grouped_results[metric_columns].sem().reset_index()
    return group_means, group_standard_errors


def generate_and_save_error_bar_plot(means: pd.DataFrame, standard_errors: pd.DataFrame, png_path: str,
                                     ylabel: str, axes_columns: List[str], axes_titles: Optional[List[str]] = None):
    """
    Plot the means with standard error bars.

    Save the plot to the given png file.

    Args:
        means: Should contain columns 'latent_dim', 'epoch' and axes_columns. A separate axes will be plotted for each
            column in axes_columns. On each axis, a separate line will be plotted for each latent dimension. Each line
            will show the values in the column plotted again the epoch for the corresponding latent dimension.
        standard_errors: The standard error for each value in means. Has the same shape and columns as means.
        png_path: The file path to save the plot as a png file.
        ylabel: The y-axis label. All plots will share the same y-axis label.
        axes_columns: The column in means to plot on each axis. A subplot will be generated with shape
            (1, len(axes_columns)).
        axes_titles: A title for each axis. Should have the same length as axes_columns. If None, will be set to
            axes_columns.
    """
    axes_titles = axes_titles or axes_columns
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(1, len(axes_columns), sharey=True, figsize=(18, 6))

    for latent_dim in means['latent_dim'].unique():
        group_means = means[means['latent_dim'] == latent_dim]
        group_standard_errors = standard_errors[standard_errors['latent_dim'] == latent_dim]

        for ax, column in zip(axes, axes_columns):
            x = group_means['epoch']
            y = group_means[column]
            se = group_standard_errors[column]
            ax.errorbar(x, y, se, label=f'latent dim = {latent_dim}', marker=None)

    for ax, title in zip(axes, axes_titles):
        ax.set_xlabel('Epoch')
        ax.set_title(title)

    axes[0].set_ylabel(ylabel)
    plt.legend()

    plt.savefig(png_path, format='png')


@click.command()
@click.option('--results-input-path', type=str, help='The parquet file path from which to load the experiment results')
@click.option('--analysis-output-dir', type=str, help='The directory path to save the output of the analysis')
def main(results_input_path: str, analysis_output_dir: str):
    """
    Analyse the results from the linear models experiments.

    Save the analysis to the given output directory.

    Args:
        results_input_path: The parquet file path from which to load the experiment results.
        analysis_output_dir: The directory path to save the output of the analysis.
    """
    results = pd.read_parquet(results_input_path)

    Path(analysis_output_dir).mkdir(parents=True, exist_ok=True)

    run_analysis(
        results,
        analysis_output_dir,
    )


if __name__ == '__main__':
    main()
