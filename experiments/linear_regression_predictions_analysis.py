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
            - pinball05_pretrained: (float) The pinball loss with alpha=0.05 of the pre-trained weight vector.
            - pinball05_swa: (float) The pinball loss with alpha=0.05 of the average weight vector (SWA solution).
            - pinball05_gradient_fa: (float) The pinball loss with alpha=0.05 of the ensemble constructed from the online
                gradient FA posterior.
            - pinball05_em_fa: (float) The pinball loss with alpha=0.05 of the ensemble constructed from the online EM FA
                posterior.
            - pinball95_pretrained: (float) The pinball loss with alpha=0.95 of the pre-trained weight vector.
            - pinball95_swa: (float) The pinball loss with alpha=0.95 of the average weight vector (SWA solution).
            - pinball95_gradient_fa: (float) The pinball loss with alpha=0.95 of the ensemble constructed from the online
                gradient FA posterior.
            - pinball95_em_fa: (float) The pinball loss with alpha=0.95 of the ensemble constructed from the online EM FA
                posterior.
            - dataset: (str) The name of the dataset.
            - fold: (int) The index of the cross-validation fold.
        analysis_output_dir: The directory path to save the output of the analysis.
    """
    metric_columns = ['mse_pretrained', 'mse_swa', 'mse_gradient_fa', 'mse_em_fa'] + \
                     ['pinball05_pretrained', 'pinball05_swa', 'pinball05_gradient_fa', 'pinball05_em_fa'] + \
                     ['pinball95_pretrained', 'pinball95_swa', 'pinball95_gradient_fa', 'pinball95_em_fa']

    for dataset_label in results['dataset'].unique():
        dataset_results = results[results['dataset'] == dataset_label]

        means = dataset_results[metric_columns].mean().reset_index()
        standard_errors = dataset_results[metric_columns].sem().reset_index()

        means.to_csv(os.path.join(
            analysis_output_dir,
            f'linear_regression_predictions_metric_means__{dataset_label}.csv'),
            index=False,
        )

        standard_errors.to_csv(os.path.join(
            analysis_output_dir,
            f'linear_regression_predictions_metric_standard_errors__{dataset_label}.csv'),
            index=False,
        )

        generate_and_save_error_bar_plot(
            results=dataset_results,
            png_path=os.path.join(
                analysis_output_dir, f'linear_regression_predictions_mse__{dataset_label}.png',
            ),
        )


def generate_and_save_error_bar_plot(results: pd.DataFrame, png_path: str):
    """
    For each algorithm, plot the mean of the MSE with standard error bars.

    Save the plot to the given png file.

    Args:
        results: Should contain columns 'mse_pretrained', 'mse_swa', 'mse_gradient_fa', 'mse_em_fa'. For each column,
            the mean and standard error will be plotted on a single figure.
        png_path: The file path to save the plot as a png file.
    """
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    metric_columns = ['mse_pretrained', 'mse_swa', 'mse_gradient_fa', 'mse_em_fa']
    labels = ['Pre-trained', 'SWA', 'SGA FA ensemble', 'EM FA Ensemble']
    markers = ['o', 's', 'v', 'X']
    for x, metric_name in enumerate(metric_columns):
        y = results[metric_name].mean()
        se = results[metric_name].sem()

        ax.errorbar(x, y, se, marker=markers[x], markersize=20, elinewidth=3, label=labels[x], capsize=10, capthick=3)

    ax.set_xticks([])
    plt.ylabel('Mean squared error')
    plt.legend()

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
