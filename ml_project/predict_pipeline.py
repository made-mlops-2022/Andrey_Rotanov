from typing import NoReturn
import logging
import pickle
import pandas as pd
import click
import sys
from data.make_dataset import read_data
from models.model_fit_predict import *
from features.build_features import *

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command()
@click.option('--path_to_model', type=click.Path(exists=True),
              default='./models/models/model.pkl',
              help='The path to model to make prediciion')
@click.option('--path_to_transformer', type=click.Path(exists=True),
              default='./models/transformers/transformer.pkl',
              help='The path to the transformer for data transformation')
@click.option('--path_to_data', type=click.Path(exists=True),
              default='./data/raw/heart_cleveland_upload.csv',
              help='Path to raw data')
@click.option('--path_to_prediction', type=click.Path(exists=False),
              default='./models/predictions/predict.csv',
              help='Path to save prediction')
def predict_pipeline(path_to_model: str, path_to_transformer: str,
                     path_to_data: str, path_to_prediction: str) -> NoReturn:
    logger.info("Start predicting")
    data = read_data(path_to_data)
    logger.info(f'Loading the transformer')

    with open(path_to_transformer, 'rb') as f:
        transformer = pickle.load(f)

    data = make_features(transformer, data)

    logger.info(f'Loading the model')
    with open(path_to_model, 'rb') as f:
        model = pickle.load(f)

    predictions = predict_model(model, data)
    logger.info(f'Made prediction using {type(model).__name__}')
    pd.DataFrame(predictions).to_csv(path_to_prediction, index=False)
    logger.info(f'Saved model to {path_to_prediction}')


if __name__ == "__main__":
    predict_pipeline()
