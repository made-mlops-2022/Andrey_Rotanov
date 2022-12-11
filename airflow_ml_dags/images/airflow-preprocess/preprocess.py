import os
import pandas as pd
import click
from sklearn.impute import SimpleImputer
import shutil


@click.command()
@click.option('--input-dir', type=click.Path(),
              default='./',
              help='The path to the data file')
@click.option('--output-dir', type=click.Path(),
              default='./',
              help='File save folder')
def preprocess(input_dir: str,
               output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    name_raw_file = "data.csv"
    name_target_file = "target.csv"
    name_processed_file = 'processed.csv'

    path_to_raw_file = os.path.join(input_dir, name_raw_file)
    if not os.path.exists(path_to_raw_file):
        raise 'File not found'

    X_data = pd.read_csv(path_to_raw_file)

    transformer = SimpleImputer(strategy='most_frequent')
    transform_data = transformer.fit_transform(X_data)
    transform_data = pd.DataFrame(transform_data, columns=X_data.columns)

    transform_data.to_csv(os.path.join(output_dir, name_processed_file), index=False)
    shutil.copy(os.path.join(input_dir, name_target_file),
                os.path.join(output_dir, name_target_file))


#
if __name__ == "__main__":
    preprocess()
