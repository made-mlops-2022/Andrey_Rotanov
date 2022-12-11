import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--input-dir', type=click.Path(),
              default='./',
              help='The path to the data file')
@click.option('--test-size', type=float,
              default=0.2,
              help='The size of the test part of the data after splitting')
@click.option('--output-dir', type=click.Path(),
              default='./',
              help='File save folder')
def split(input_dir: str,
                test_size: float,
                output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    name_processed_file_val = 'X_val.csv'
    name_processed_file_train = 'X_train.csv'
    name_target_file_val = 'y_val.csv'
    name_target_file_train = 'y_train.csv'

    name_process_file = 'processed.csv'
    name_target_file = 'target.csv'
    path_to_preprocess_file = os.path.join(input_dir, name_process_file)
    path_to_target_file = os.path.join(input_dir, name_target_file)
    if not os.path.exists(path_to_preprocess_file):
        raise 'File not found'

    transform_data = pd.read_csv(path_to_preprocess_file)
    y_data = pd.read_csv(path_to_target_file)
    X_train, X_val, y_train, y_val = train_test_split(transform_data,
                                                      y_data,
                                                      test_size=test_size)

    X_train.to_csv(os.path.join(output_dir, name_processed_file_train), index=False)
    X_val.to_csv(os.path.join(output_dir, name_processed_file_val), index=False)
    y_train.to_csv(os.path.join(output_dir, name_target_file_train), index=False)
    y_val.to_csv(os.path.join(output_dir, name_target_file_val), index=False)

if __name__=="__main__":
    split()