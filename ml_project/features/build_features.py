from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from enities import FeatureParams
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def extract_target(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    return data.drop(target_column, axis=1), data[target_column]


def drop_columns(data: pd.DataFrame, columns_to_delete: list) -> pd.DataFrame:
    return data.drop(columns_to_delete, axis=1)


def create_categorical_pipeline(to_process=True) -> Pipeline:
    if to_process:
        return Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder())])
    else:
        return Pipeline([('impute', SimpleImputer(strategy='most_frequent'))])


def create_numerical_pipeline(to_process=True) -> Pipeline:
    if to_process:
        return Pipeline([('impute', SimpleImputer(strategy='mean')),
                         ('scaler', StandardScaler())])
    else:
        return Pipeline([('impute', SimpleImputer(strategy='mean'))])


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def create_transformer(features_params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                create_categorical_pipeline(features_params.process_categorical),
                np.array(features_params.categorical_features)
            ),
            (
                'numerical_pipeline',
                create_numerical_pipeline(features_params.process_numerical),
                np.array(features_params.numerical_features),
            ),
        ]
    )
    return transformer
