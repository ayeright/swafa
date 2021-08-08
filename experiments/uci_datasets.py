import os
from pathlib import Path

import pandas as pd
import click


def download_all_datasets() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Download datasets from the UCI Machine Learning Repository.

    Target variable of each dataset is in the final column and all other columns are features.

    Returns:
        boston_housing_data: Boston Housing dataset.
        yacht_hydrodynamics_data: Yacht Hydrodynamics dataset.
        concrete_strength_data: Concrete Compressive Strength dataset.
        energy_efficiency_data: Energy Efficiency dataset.
    """
    boston_housing_data = download_boston_housing_dataset()
    yacht_hydrodynamics_data = download_yacht_hydrodynamics_dataset()
    concrete_strength_data = download_concrete_strength_dataset()
    energy_efficiency_data = download_energy_efficiency_dataset()
    return boston_housing_data, yacht_hydrodynamics_data, concrete_strength_data, energy_efficiency_data


def download_boston_housing_dataset() -> pd.DataFrame:
    """
    Download the Boston Housing dataset from the UCI Machine Learning Repository.

    Returns:
        Boston Housing dataset. Target variable is in final column.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    columns = [
        'CRIM',
        'ZN',
        'INDUS',
        'CHAS',
        'NOX',
        'RM',
        'AGE',
        'DIS',
        'RAD',
        'TAX',
        'PTRATIO',
        'B',
        'LSTAT',
        'MEDV',
    ]
    return pd.read_csv(url, delim_whitespace=True, names=columns)


def download_yacht_hydrodynamics_dataset() -> pd.DataFrame:
    """
    Download the Yacht Hydrodynamics dataset from the UCI Machine Learning Repository.

    Returns:
        Yacht Hydrodynamics dataset. Target variable is in final column.
    """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
    columns = [
        'buoyancy_longitudinal_position',
        'prismatic_coefficient',
        'length_displacement_ratio',
        'beam_draught_ratio',
        'length_beam_ratio',
        'froude_number',
        'residuary_resistance',
    ]
    return pd.read_csv(url, delim_whitespace=True, names=columns)


def download_concrete_strength_dataset() -> pd.DataFrame:
    """
    Download the Concrete Compressive Strength dataset from the UCI Machine Learning Repository.

    Returns:
        Concrete Compressive Strength dataset. Target variable is in final column.
    """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    return pd.read_excel(url)


def download_energy_efficiency_dataset() -> pd.DataFrame:
    """
    Download the Energy Efficiency dataset from the UCI Machine Learning Repository.

    Returns:
        Energy Efficiency dataset. Target variable is in final column. This dataset has two targets, heating and cooling
            load, but we return only heating load.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
    return pd.read_excel(url).iloc[:, :-1]


@click.command()
@click.option('--boston-housing-output-path', type=str, help='The parquet file path to save the Boston Housing dataset')
@click.option('--yacht-hydrodynamics-output-path', type=str, help='The parquet file path to save the Yacht '
                                                                  'Hydrodynamics dataset')
@click.option('--concrete-strength-output-path', type=str, help='The parquet file path to save the Concrete '
                                                                'Compressive Strength dataset')
@click.option('--energy-efficiency-output-path', type=str, help='The parquet file path to save the Energy Efficiency '
                                                                'dataset')
def main(boston_housing_output_path: str, yacht_hydrodynamics_output_path: str, concrete_strength_output_path: str,
         energy_efficiency_output_path: str):
    """
    Download datasets from the UCI Machine Learning Repository and save them to disk.

    Args:
        boston_housing_output_path: The parquet file path to save the Boston Housing dataset.
        yacht_hydrodynamics_output_path: The parquet file path to save the Yacht Hydrodynamics dataset.
        concrete_strength_output_path: The parquet file path to save the Concrete Compressive Strength dataset.
        energy_efficiency_output_path: The parquet file path to save the Energy Efficiency dataset.
    """
    boston_housing_data, yacht_hydrodynamics_data, concrete_strength_data, energy_efficiency_data \
        = download_all_datasets()

    for x in [boston_housing_output_path, yacht_hydrodynamics_output_path, concrete_strength_output_path,
              energy_efficiency_output_path]:
        Path(os.path.dirname(x)).mkdir(parents=True, exist_ok=True)

    boston_housing_data.to_parquet(boston_housing_output_path)
    yacht_hydrodynamics_data.to_parquet(yacht_hydrodynamics_output_path)
    concrete_strength_data.to_parquet(concrete_strength_output_path)
    energy_efficiency_data.to_parquet(energy_efficiency_output_path)


if __name__ == '__main__':
    main()
