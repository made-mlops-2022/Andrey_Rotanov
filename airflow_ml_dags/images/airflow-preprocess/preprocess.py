import os
import pandas as pd
import click
from sklearn.impute import SimpleImputer
import shutil
import logging

logger = logging.getLogger(__name__)
_log_format = "%(asctime)s\t%(levelname)s\t %(message)s"
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)
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
    logger.info('Preprocess reading file')
    X_data = pd.read_csv(path_to_raw_file)
    logger.info('Preprocess TRANSFORM file')
    transformer = SimpleImputer(strategy='most_frequent')
    transform_data = transformer.fit_transform(X_data)
    transform_data = pd.DataFrame(transform_data, columns=X_data.columns)
    logger.info('Preprocess SAVE file')
    transform_data.to_csv(os.path.join(output_dir, name_processed_file), index=False)
    shutil.copy(os.path.join(input_dir, name_target_file),
                os.path.join(output_dir, name_target_file))
    logger.info(f'Preprocess FINISHED {os.path.join(output_dir, name_processed_file)}\n'
                f' {os.path.join(input_dir, name_target_file)}\n'
                f' {os.path.join(input_dir, name_target_file)}')
    # y_data = pd.read_csv(path_to_target_file)
    #
    # X_train, X_val, y_train, y_val = train_test_split(transform_data,
    #                                                   y_data,
    #                                                   test_size=test_size)
    # X_train.to_csv(os.path.join(output_dir, name_processed_file_train), index=False)
    # X_val.to_csv(os.path.join(output_dir, name_processed_file_val), index=False)
    # y_train.to_csv(os.path.join(output_dir, name_target_file_train), index=False)
    # y_val.to_csv(os.path.join(output_dir, name_target_file_val), index=False)

#
if __name__ == "__main__":
    preprocess()
