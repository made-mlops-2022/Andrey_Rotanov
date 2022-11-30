from typing import NoReturn
import click
import pandas as pd
from faker import Faker
import os
import logging

logger = logging.getLogger(__name__)
_log_format = "%(asctime)s\t%(levelname)s\t %(message)s"
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

def generate_synthetic_data(cnt_rows: int) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(21)
    df = {
        "age": [fake.pyint(min_value=25, max_value=80) for _ in range(cnt_rows)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(cnt_rows)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(cnt_rows)],
        "chol": [fake.pyint(min_value=126, max_value=555) for _ in range(cnt_rows)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(cnt_rows)],
        "thalach": [fake.pyint(min_value=71, max_value=202) for _ in range(cnt_rows)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(cnt_rows)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(cnt_rows)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(cnt_rows)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(cnt_rows)],
        "condition": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],

    }
    return pd.DataFrame.from_dict(df)


@click.command()
@click.option('--output-dir', type=click.Path(),
              default='./',
              help='The way to save the generated data')
@click.option('--size', type=int,
              default=500,
              help='The size of the generated data')
@click.option('--name-data-file', type=str,
              default='data.csv',
              help='Name of the generated file')
@click.option('--name-target-file', type=str,
              default='target.csv',
              help='The name of the file with class labels')
def generate(output_dir: str,
             size: int,
             name_data_file: str,
             name_target_file: str) -> NoReturn:

    os.makedirs(output_dir, exist_ok=True)
    target_column = 'condition'
    logger.info(f'Cur dir {os.getcwd()}')
    path_to_data_file = os.path.join(output_dir, name_data_file)
    logger.info(f'{path_to_data_file=}')
    if os.path.exists(path_to_data_file):
        os.remove(path_to_data_file)

    path_to_target_file = os.path.join(output_dir, name_target_file)
    if os.path.exists(path_to_target_file):
        os.remove(path_to_target_file)

    df = generate_synthetic_data(size)

    data = pd.DataFrame(data=df)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X.to_csv(path_to_data_file, index=False)
    y.to_csv(path_to_target_file, index=False)
#

if __name__ == "__main__":
    generate()
