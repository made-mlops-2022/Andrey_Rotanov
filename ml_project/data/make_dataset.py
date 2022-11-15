from typing import Tuple, NoReturn
import os
import pandas as pd
import numpy as np
from marshmallow_dataclass import class_schema
from sklearn.model_selection import train_test_split
from enities import DownloadParams, SplittingParams
from enities import SyntheticDataParams, DescriptionFeature
import hydra
from faker import Faker


def download_data_from_kaggle(params: DownloadParams) -> NoReturn:
    os.environ['KAGGLE_USERNAME'] = params.username
    os.environ['KAGGLE_KEY'] = params.api_key
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(params.dataset_name, path=params.output_folder, unzip=True)


def read_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def split_train_val_data(X: pd.DataFrame,
                         y: pd.Series,
                         params: SplittingParams) -> Tuple[pd.DataFrame,
                                                           pd.DataFrame,
                                                           pd.Series,
                                                           pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=params.val_size,
                                                        random_state=params.random_state)
    return X_train, X_test, y_train, y_test


def generate_synthetic_data(cnt_rows: int) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(21)
    df = {
        "age": [fake.pyint(min_value=25, max_value=80) for _ in range(cnt_rows)],
        "sex": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],
        "cp": [fake.pyint(min_value=0, max_value=3) for _ in range(cnt_rows)],
        "trestbps": [fake.pyint(min_value=94, max_value=200) for _ in range(cnt_rows)],
        "chol": [fake.pyint(min_value=126, max_value=555) for _ in range(cnt_rows)],
        "fbs": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],
        "restecg": [fake.pyint(min_value=0, max_value=2) for _ in range(cnt_rows)],
        "thalach": [fake.pyint(min_value=71, max_value=202) for _ in range(cnt_rows)],
        "exang": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],
        "oldpeak": [fake.pyfloat(min_value=0, max_value=7) for _ in range(cnt_rows)],
        "slope": [fake.pyint(min_value=0, max_value=2) for _ in range(cnt_rows)],
        "ca": [fake.pyint(min_value=0, max_value=4) for _ in range(cnt_rows)],
        "thal": [fake.pyint(min_value=0, max_value=3) for _ in range(cnt_rows)],
        "condition": [fake.pyint(min_value=0, max_value=1) for _ in range(cnt_rows)],

    }
    return pd.DataFrame.from_dict(df)


@hydra.main(version_base=None, config_path='../configs', config_name='synthetic_data_config')
def generate_config(params: SyntheticDataParams) -> NoReturn:
    fake = Faker()
    Faker.seed(21)
    df = {}
    for cur_feature in params.feature:
        DescriptionFeatureSchema = class_schema(DescriptionFeature)
        schema = DescriptionFeatureSchema()
        feature = schema.load(cur_feature['DescriptionFeature'])
        if feature.type == "int":
            df[feature.name] = [fake.pyint(min_value=feature.min_value,
                                           max_value=feature.max_value) for _ in range(params.size)].copy()
    pd.DataFrame(data=df).to_csv(params.output_path)
