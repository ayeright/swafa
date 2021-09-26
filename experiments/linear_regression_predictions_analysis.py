import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import click


def run_analysis(results: pd.DataFrame, analysis_output_dir: str):
    """
    Aggregate the experiment results and generate box plots to show the distribution of the test mean squared error for
    each dataset.

    Args:
        results: The results of each cross-validation fold. The number of rows in the DataFrame is equal to
            n_datasets * n_folds. The DataFrame has the following columns:
            - mse_pretrained: (float) The test MSE of the pre-trained weight vector.
            - mse_swa: (float) The test MSE of the average weight vector (SWA solution).
            - mse_gradient_fa: (float) The test MSE of the ensemble constructed from the online gradient FA posterior.
            - mse_em_fa: (float) The test MSE of the ensemble constructed from the online EM FA posterior.
            - dataset: (str) The name of the dataset.
            - fold: (int) The index of the cross-validation fold.
        analysis_output_dir: The directory path to save the output of the analysis.
    """
    plt.rcParams.update({'font.size': 15})

    metric_columns = ['mse_pretrained', 'mse_swa', 'mse_gradient_fa', 'mse_em_fa']

    for dataset_label in results['dataset'].unique():
        dataset_results = results[results['dataset'] == dataset_label]

        grouped_results = dataset_results.groupby('fold')
        group_means = grouped_results[metric_columns].mean().reset_index()
        group_standard_errors = grouped_results[metric_columns].sem().reset_index()

        group_means.to_csv(os.path.join(
            analysis_output_dir,
            f'linear_regression_predictions_metric_means__{dataset_label}.csv'),
            index=False,
        )

        group_standard_errors.to_csv(os.path.join(
            analysis_output_dir,
            f'linear_regression_predictions_metric_standard_errors__{dataset_label}.csv'),
            index=False,
        )

        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        dataset_results.boxplot(column=['mse_pretrained', 'mse_swa', 'mse_gradient_fa', 'mse_em_fa'], grid=True)
        ax.set_xticklabels(['pre-trained', 'SWA', 'online SGA ensemble', 'online EM ensemble'])
        plt.ylabel('Mean squared error')

        png_path = os.path.join(analysis_output_dir, f'linear_regression_predictions_mse__{dataset_label}.png')
        plt.savefig(png_path, format='png')


@click.command()
@click.option('--results-input-path', type=str, help='The parquet file path from which to load the experiment results')
@click.option('--analysis-output-dir', type=str, help='The directory path to save the output of the analysis')
def main(results_input_path: str, analysis_output_dir: str):
    """
    Analyse the results from the linear regression predictions experiments.

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
