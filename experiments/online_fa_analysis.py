import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click
import yaml


def run_analysis(results: pd.DataFrame, analysis_output_dir: str, min_samples: int):
    """
    Aggregate the experiment results and generate plots for covariance distance and log-likelihood.

    For each experiment, defined by the parameters observation_dim, latent_dim, spectrum_min and spectrum_max, group by
    n_samples and compute the mean and standard error of the metrics in the experiment results. Save these statistics
    to csv files.

    Also, for each experiment generate two plots, one showing the distance between the true covariance matrix and the
    estimated covariance matrix and another showing the log-likelihood of the model, for each factor analysis (FA)
    learning algorithm. Save these plots to png files.

    Args:
        results: The results of each experiment. Has len(experiments_config) * n_trials rows and the following columns:
            - observation_dim: (int) Same as above.
            - latent_dim: (int) Same as above.
            - spectrum_min: (float) The lower bound of the spectrum range.
            - spectrum_max: (float) The upper bound of the spectrum range.
            - n_samples: (int) Same as above.
            - covar_norm: (float) The Frobenius norm of the the true covariance matrix of the FA model.
            - covar_distance_sklearn: (float) The Frobenius norm of the difference between the true covariance matrix
                and the covariance matrix estimated by sklearn's `FactorAnalysis`.
            - covar_distance_online_gradient: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineGradientFactorAnalysis`.
            - covar_distance_online_em: (float) The Frobenius norm of the difference between the true covariance
                matrix and the covariance matrix estimated by `OnlineEMFactorAnalysis`.
            - ll_true: (float) The log-likelihood of the true FA model, given the data sampled from the model.
            - ll_sklearn: (float) The log-likelihood of the sklearn FA model.
            - ll_online_gradient: (float) The log-likelihood of the online gradient FA model.
            - ll_online_em: (float) The log-likelihood of the online EM FA model.
            - experiment: (int) The index of the experiment.
            - trial: (int) The index of the trial within the experiment.
        analysis_output_dir: The directory path to save the output of the analysis.
        min_samples: Only analyse experiments which used at least this many data samples to learn FA models.
    """
    param_columns = ['observation_dim', 'latent_dim', 'spectrum_min', 'spectrum_max']
    group_by_column = 'n_samples'
    covar_columns = [
        'covar_norm', 'covar_distance_sklearn', 'covar_distance_online_gradient', 'covar_distance_online_em'
    ]
    ll_columns = ['ll_true', 'll_sklearn', 'll_online_gradient', 'll_online_em']

    results = results[results['n_samples'] >= min_samples]

    param_combinations = results[param_columns].drop_duplicates()
    for _, params in param_combinations.iterrows():
        group_means, group_standard_errors = aggregate_experiment_results(
            results, params, group_by_column, covar_columns + ll_columns,
        )

        params_str = params_to_string(params)

        group_means.to_csv(os.path.join(
            analysis_output_dir, f'online_fa_metric_means__{params_str}.csv'),
        )

        group_standard_errors.to_csv(os.path.join(
            analysis_output_dir, f'online_fa_metric_standard_errors__{params_str}.csv'),
        )

        generate_and_save_error_bar_plot(
            group_means[covar_columns],
            group_standard_errors[covar_columns],
            png_path=os.path.join(analysis_output_dir, f'online_fa_covar_distance__{params_str}.png'),
            xlabel='Number of samples',
            ylabel='Frobenius norm',
            xscale='log',
            yscale='log',
        )

        generate_and_save_error_bar_plot(
            group_means[ll_columns],
            group_standard_errors[ll_columns],
            png_path=os.path.join(analysis_output_dir, f'online_fa_log_likelihood__{params_str}.png'),
            xlabel='Number of samples',
            ylabel='Log-likelihood',
            xscale='log',
            yscale='linear',
        )


def aggregate_experiment_results(results: pd.DataFrame, experiment_params: Union[pd.Series, dict],
                                 group_by_column: str, metric_columns: List[str]) -> (pd.DataFrame, pd.DataFrame):
    """
    For the experiment with the given parameters, group by the given column and average the given metric columns over
    all trials.

    Also, compute the standard error of the mean of each group.

    Args:
        results: The results of each experiment. Each row corresponds to a single trial. Columns must contain
            group_by_column, metric_columns and all keys in experiment_params.
        experiment_params: The parameters of the experiment which is to be aggregated. Keys are parameter names and
            values are parameter values.
        group_by_column: The column to group by before aggregating the results.
        metric_columns: The columns which are to be aggregated.

    Returns:
        group_means: The mean of each column in metric_columns for each experiment group. The number of rows is equal to
            the number of unique values in group_by_column and the number of columns is equal to len(metric_columns).
        group_standard_errors: The standard error of each value in group_means. Has the same shape as group_means.
    """
    experiment_mask = np.all(
        np.hstack(
            [results[name].values.reshape(-1, 1) == value for name, value in experiment_params.items()]
        ),
        axis=1
    )
    experiment_results = results[experiment_mask]
    grouped_results = experiment_results.groupby(group_by_column)
    group_means = grouped_results[metric_columns].mean()
    group_standard_errors = grouped_results[metric_columns].sem()
    return group_means, group_standard_errors


def generate_and_save_error_bar_plot(means: pd.DataFrame, standard_errors: pd.DataFrame, png_path: str, xlabel: str,
                                     ylabel: str, xscale: str = 'linear', yscale: str = 'linear'):
    """
    Plot the means with standard error bars.

    Use a log scale for both axes.

    Save the plot to the given png file.

    Args:
        means: Plot a separate line for the values in each column. Use the values in the index of the DataFrame as the
            x-axis values.
        standard_errors: The standard error for each value in means. Has the same shape as means.
        png_path: The file path to save the plot as a png file.
        ylabel: The x-axis label.
        ylabel: The y-axis label.
        xscale: The type of scale to use on the x-axis.
        yscale: The type of scale to use on the y-axis.
    """
    plt.figure(figsize=(18, 8))
    x = means.index
    for metric_name in means.columns:
        plt.errorbar(x, means[metric_name], standard_errors[metric_name], label=metric_name, marker='o')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.legend()

    plt.savefig(png_path, format='png')


def params_to_string(params: Union[pd.Series, dict]) -> str:
    """
    Convert parameter key-value pairs to a string.

    E.g. dict(observation_dim=19, latent_dim=5) will be converted to 'observation_dim=19__latent_dim=5'.

    Args:
        params: Key-value parameter pairs.

    Returns:
        A string of the form 'key1=value1__key2==value2__.....'.
    """
    return '__'.join([f'{name}={value}' for name, value in params.items()])


@click.command()
@click.option('--results-input-path', type=str, help='The parquet file path from which to load the experiment results')
@click.option('--analysis-output-dir', type=str, help='The directory path to save the output of the analysis')
def main(results_input_path: str, analysis_output_dir: str):
    """
    Analyse the results from the factor analysis experiments.

    Save the analysis to the given output directory.

    Args:
        results_input_path: The parquet file path from which to load the experiment results.
        analysis_output_dir: The directory path to save the output of the analysis.
    """
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    results = pd.read_parquet(results_input_path)

    Path(analysis_output_dir).mkdir(parents=True, exist_ok=True)

    run_analysis(
        results,
        analysis_output_dir,
        params['online_fa_analysis']['min_samples'],
    )


if __name__ == '__main__':
    main()
