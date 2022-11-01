from typing import NoReturn
import logging
import pickle
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import pandas as pd
import click
import mlflow
import json
from enities.train_pipeline_params import TrainingPipelineParams
import sys
from data.make_dataset import *
from models.model_fit_predict import *
from features.build_features import *

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# @click.command()
# @click.option('--path_to_model', type=click.Path(exists=True),
#               default='models/model_knn.pkl',
#               help='The path to model to make prediciion')
# @click.option('--path_to_transformer', type=click.Path(exists=True),
#               default='models/transformers/transformer_knn.pkl',
#               help='The path to the transformer for data transformation')
# @click.option('--path_to_data', type=click.Path(exists=True),
#               default='data/test/heart_cleveland_upload_test_unlabeled.csv',
#               help='Path to raw data')
# @click.option('--path_to_prediction', type=click.Path(exists=False),
#               default='models/predictions/pred_knn.csv',
#               help='Path to save prediction')
@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train_pipeline(params: TrainingPipelineParams) -> NoReturn:
    # params = instantiate(params, _convert_='partial')
    logger.info("Reading data")
    data = read_data(params.input_data_path)

    X, y = extract_target(data, params.feature_params.target_col)
    X_train, X_test, y_train, y_test = split_train_val_data(X, y, params.splitting_params)
    logger.info(f'Test size: {len(X_test)}')
    logger.info(f'Train size: {len(X_train)}')

    logger.info(f'Create transformer')
    transformer = create_transformer(params.feature_params)
    logger.info(f'Transformer training')
    transformer.fit(X_train)
    logger.info(f'Save transformer {params.train_params.output_transformer_path}')
    serialize_model(transformer, params.train_params.output_transformer_path)

    logger.info(f'Start transformer')
    X_train = make_features(transformer, X_train)

    with open(params.train_params.output_transformer_path, 'rb') as f:
        transformer = pickle.load(f)

    if params.use_mlflow:
        mlflow.set_tracking_uri("http://localhost:5000")
        with mlflow.start_run(run_name=params.name_training_in_mlflow):
            logger.info(f'Start training {params.train_params.model_type}...')
            model = train_model(X_train, y_train, params.train_params)
            logger.info('End training')

            inference_pipeline = create_inference_pipeline(model, transformer)
            y_pred = predict_model(inference_pipeline, X_test)

            logger.info('Evaluate models')
            metrics = evaluate_model(y_pred, y_test)
            for metric in metrics:
                mlflow.log_metric(metric, metrics[metric])

            logger.info("Save metrics")
            with open(params.train_params.output_metric_path, 'w') as file:
                json.dump(metrics, file)

            logger.info(f"Save model to {params.train_params.output_model_path}")
            serialize_model(model, params.train_params.output_model_path)

            # mlflow.sklearn.log_model(
            #     sk_model=model,
            #     artifact_path="classification_model",
            #     registered_model_name=params.train_params.model_type)


if __name__ == "__main__":
    train_pipeline()
