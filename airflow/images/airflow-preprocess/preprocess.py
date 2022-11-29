import os
import pandas as pd
import click
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--input-dir', type=click.Path(),
              default='./',
              help='The path to the data file')
@click.option('--path-target-file', type=click.Path(),
              default='./',
              help='The path to the data file')
@click.option('--test-size', type=float,
              default=0.2,
              help='The size of the test part of the data after splitting')
@click.option('--output-dir', type=click.Path(),
              default='./',
              help='File save folder')
def preprocess(path_to_raw_file: str,
                    path_to_target_file: str,
                    test_size: float,
                    output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    name_processed_file_val = 'X_val.csv'
    name_processed_file_train = 'X_train.csv'
    name_target_file_val = 'y_val.csv'
    name_target_file_train = 'y_train.csv'
    if not os.path.exists(path_to_raw_file):
        raise 'File not found'

    X_data = pd.read_csv(path_to_raw_file)
    transformer = SimpleImputer(strategy='most_frequent')
    transform_data = transformer.fit_transform(X_data)
    transform_data = pd.DataFrame(transform_data, columns=X_data.columns)

    y_data = pd.read_csv(path_to_target_file)

    X_train, X_val, y_train, y_val = train_test_split(transform_data,
                                                      y_data,
                                                      test_size=test_size)
    X_train.to_csv(os.path.join(output_dir, name_processed_file_train), index=False)
    X_val.to_csv(os.path.join(output_dir, name_processed_file_val), index=False)
    y_train.to_csv(os.path.join(output_dir, name_target_file_train), index=False)
    y_val.to_csv(os.path.join(output_dir, name_target_file_val), index=False)


if __name__ == "__main__":
    preprocess()
