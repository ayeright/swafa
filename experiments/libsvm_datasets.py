import os
from pathlib import Path
import tempfile
from typing import Optional
import urllib

import click
import pandas as pd
from sklearn.datasets import load_svmlight_file


AUSTRALIAN_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale'
BREAST_CANCER_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale'


def download_all_datasets() -> (pd.DataFrame, pd.DataFrame):
    """
    Download datasets from the LIBSVM repository.

    Binary target variable of each dataset is in the final column and all other columns are features.

    Returns:
        australian_data: Scaled Australian dataset.
        breast_cancer_data: Scaled Breast Cancer dataset.
    """
    australian_data = download_libsvm_dataset(url=AUSTRALIAN_URL, y_map={-1: 0})
    breast_cancer_data = download_libsvm_dataset(url=BREAST_CANCER_URL, y_map={2: 0, 4: 1})
    return australian_data, breast_cancer_data


def download_libsvm_dataset(url: str, y_map: Optional[dict] = None) -> pd.DataFrame:
    """
    Download the LIBSVM dataset at the given URL.

    Target variable is in the final column and all other columns are features.

    Args:
        url: The URL of the dataset.
        y_map: Map for substituting values of the target variable.

    Returns:
        A DataFrame with columns x1, x2, ..., xn, y.
    """
    y_map = y_map or dict()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_file_path = os.path.join(tmp, 'libsvm.txt')
        urllib.request.urlretrieve(url, tmp_file_path)
        X, y = load_svmlight_file(tmp_file_path)

    df = pd.DataFrame(X.toarray(), columns=[f'x{i}' for i in range(X.shape[1])])
    df['y'] = y
    df['y'] = df['y'].replace(y_map)

    return df


@click.command()
@click.option('--australian-output-path', type=str, help='The parquet file path to save the scaled Australian dataset')
@click.option('--breast-cancer-output-path', type=str, help='The parquet file path to save the scaled Breast Cancer '
                                                            'dataset')
def main(australian_output_path: str, breast_cancer_output_path: str):
    """
    Download datasets from the LIBSVM repository and save them to disk.

    Args:
        australian_output_path: The parquet file path to save the Australian dataset.
        breast_cancer_output_path: The parquet file path to save the Breast Cancer dataset.
    """
    australian_data, breast_cancer_data = download_all_datasets()

    for x in [australian_output_path, breast_cancer_output_path]:
        Path(os.path.dirname(x)).mkdir(parents=True, exist_ok=True)

    australian_data.to_parquet(australian_output_path)
    breast_cancer_data.to_parquet(breast_cancer_output_path)


if __name__ == '__main__':
    main()
