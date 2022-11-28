from typing import NoReturn
import pickle
import pandas as pd
import click
import os
import mlflow


@click.command()
@click.option('--path-to-data', type=click.Path(),
              default='./data/raw/',
              help='Path to input data')
@click.option('--output-dir', type=click.Path(),
              default='./',
              help='Path to save prediction')
def predict(path_to_data: str,
            output_dir: str) -> NoReturn:
    os.makedirs(output_dir, exist_ok=True)
    URL = "http://localhost:5000"
    data = pd.read_csv(os.path.join(path_to_data, "X_val.csv"))

    mlflow.set_tracking_uri(URL)
    model = mlflow.pyfunc.load_model(
        model_uri='models:/knn_model/Production'
    )

    predictions = model.predict(data)
    path_to_prediction = os.path.join(output_dir, 'predictions.csv')
    pd.DataFrame(predictions).to_csv(path_to_prediction, index=False)


if __name__ == "__main__":
    predict()
