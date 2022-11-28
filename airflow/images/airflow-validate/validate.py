import os
import json
import click
import mlflow
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score


@click.command()
@click.option('--input-dir', type=click.Path(exists=True),
              default='./',
              help='The path to the data file')
@click.option('--model-dir', type=click.Path(exists=True),
              default='./',
              help='File save folder')
@click.option('--output-dir', type=click.Path(exists=True),
              default='./',
              help='The path to the data file')
def validate(input_dir: str,
             model_dir: str,
             output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    URL = "http://localhost:5000"
    model_name = os.listdir(model_dir)[0]
    mlflow.set_tracking_uri(URL)
    with mlflow.start_run(run_name=os.path.basename(model_name)[0]):
        X = pd.read_csv(os.path.join(input_dir, 'X_val.csv'))
        y_true = pd.read_csv(os.path.join(input_dir, 'y_val.csv'))

        with open(os.path.join(model_dir, model_name), 'rb') as model_file:
            model = pickle.load(model_file)
        y_pred = model.predict(X)

        metrics = {}
        metrics["r2_score"] = r2_score(y_true, y_pred),
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False),
        metrics["mae"] = mean_absolute_error(y_true, y_pred),
        metrics['recall'] = recall_score(y_true, y_pred)

        for metric in metrics:
            mlflow.log_metric(metric, metrics[metric])

        with open(os.path.join(output_dir, 'metric.json'), 'w') as metric_file:
            json.dump(metrics, metric_file)


if __name__ == "__main__":
    validate()
