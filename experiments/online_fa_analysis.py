import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click


def run_analysis(results: pd.DataFrame, analysis_output_dir: str):
    param_columns = ['observation_dim', 'latent_dim', 'spectrum_min', 'spectrum_max']
    group_by_column = 'n_samples'
    metric_columns = ['covar_norm', 'covar_distance_sklearn', 'covar_distance_online_gradient',
                      'covar_distance_online_em']

    param_combinations = results[param_columns].drop_duplicates()
    for _, params in param_combinations.iterrows():
        group_means, group_standard_errors = aggregated_experiment_results(
            results, params, group_by_column, metric_columns,
        )

        plot_and_save_results(group_means, group_standard_errors, params, analysis_output_dir)


def aggregated_experiment_results(results: pd.DataFrame, experiment_params: pd.Series, group_by_column: str,
                                  metric_columns: List[str]) -> (pd.DataFrame, pd.DataFrame):
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


def plot_and_save_results(group_means: pd.DataFrame, group_standard_errors: pd.DataFrame, experiment_params: pd.Series,
                          output_dir: str):
    plt.figure(figsize=(18, 8))
    x = group_means.index
    for metric_name in group_means.columns:
        plt.errorbar(x, group_means[metric_name], group_standard_errors[metric_name], label=metric_name, marker='o')

    plt.xlabel('Number of samples')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    param_str = '__'.join([f'{name}={value}' for name, value in experiment_params.items()])
    file_path = os.path.join(output_dir, f'online_fa__{param_str}.png')
    plt.savefig(file_path, format='png')


@click.command()
@click.option('--results-input-path', type=str, help='The parquet file path from which to load the experiment results')
@click.option('--analysis-output-dir', type=str, help='The directory path to save the output of the analysis')
def main(results_input_path: str, analysis_output_dir: str):
    """
    Analyse the results from the factor analysis experiments.

    Args:
        results_input_path: The parquet file path from which to load the experiment results.
        analysis_output_dir: The directory path to save the output of the analysis.
    """
    results = pd.read_parquet(results_input_path)
    Path(analysis_output_dir).mkdir(parents=True, exist_ok=True)
    run_analysis(results, analysis_output_dir)


if __name__ == '__main__':
    main()
