import os
from typing import NoReturn
import pickle
import mlflow
from sklearn.neighbors import KNeighborsClassifier
import click
import pandas as pd


@click.command()
@click.option('--input-dir', type=click.Path(exists=True),
              default='data.csv',
              help='The path to the data file')
@click.option('--output-dir', type=click.Path(),
              default='./',
              help='File save folder')
def train_model(input_dir: str,
                output_dir: str) -> NoReturn:

    os.makedirs(output_dir, exist_ok=True)
    URL = "http://localhost:5000"
    mlflow.set_tracking_uri(URL)
    with mlflow.start_run(run_name='train'):
        id_run = mlflow.active_run()
        X = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
        y = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))

        model = KNeighborsClassifier(
            n_neighbors=3
        )
        model.fit(X, y)
        model_params = model.get_params()
        for param in model_params:
            mlflow.log_param(param, model_params[param])

        with open(os.path.join(output_dir, f'knn_model_{id_run.info.run_id}.pkl'), 'wb') as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classification_model",
            registered_model_name='knn_model')


if __name__ == "__main__":
    train_model()
