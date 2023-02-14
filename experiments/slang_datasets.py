import io
import requests
import zipfile

import click


@click.command()
@click.option('--datasets-output-dir', type=str, help='The directory to save SLANG datasets')
def main(datasets_output_dir: str):
    """
    Download datasets used in the SLANG paper (https://arxiv.org/pdf/1811.04504.pdf).

    Args:
        datasets_output_dir: The directory to save SLANG datasets.
    """
    zip_file_url = 'https://github.com/aaronpmishkin/SLANG/releases/download/v1.0.0/data.zip'

    r = requests.get(zip_file_url)

    if not r.ok:
        raise IOError(f'no zip file found at {zip_file_url}')

    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(datasets_output_dir)


if __name__ == '__main__':
    main()
