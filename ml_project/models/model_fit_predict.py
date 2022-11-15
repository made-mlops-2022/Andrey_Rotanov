import pickle
from typing import Dict, Union

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from enities.training_params import TrainingParams

SklearnClassificationModel = Union[KNeighborsClassifier, LogisticRegression]


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params: TrainingParams) -> SklearnClassificationModel:
    if train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=3
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def create_inference_pipeline(model: SklearnClassificationModel,
                              transformer: ColumnTransformer) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.array:
    return model.predict(features)


def evaluate_model(y_pred: np.array, y_true: np.array) -> Dict[str, float]:
    return {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }


def serialize_model(model: object, output_path: str) -> str:
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    return output_path
