import os
import sys
import unittest

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'ml_project'
)
sys.path.append(SOURCE_PATH)
import numpy as np
from data.make_dataset import (
    generate_synthetic_data,
    split_train_val_data)
from enities.train_pipeline_params import read_training_pipeline_params
from features.build_features import (
    extract_target,
    create_transformer,
    make_features)
from models.model_fit_predict import (
    train_model,
    create_inference_pipeline)


class TestFeatures(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config_path = './tests/configs/test_configs.yaml'
        params = read_training_pipeline_params(config_path)

        self.size_generate_data = (540, 14)
        self.size_test = self.size_generate_data[0] * params.splitting_params.val_size
        self.data_gen = generate_synthetic_data(self.size_generate_data[0])

        X, y = extract_target(self.data_gen, params.feature_params.target_col)
        X_train, X_test, y_train, self.y_test = split_train_val_data(X, y, params.splitting_params)
        self.transformer = create_transformer(params.feature_params)

        self.transformer.fit(X_train)
        model = train_model(X_train, y_train, params.train_params)
        self.inference_pipeline = create_inference_pipeline(model, self.transformer)
        self.X_train = make_features(self.transformer, X_train)
        self.X_test = make_features(self.transformer, X_test)

    def test_create_pipeline(self):
        self.assertEqual(len(self.inference_pipeline), 2)

    def test_process_features(self):
        self.assertEqual(self.X_train.shape, (self.size_generate_data[0] - self.size_test, 30))
        self.assertEqual(self.X_test.shape, (self.size_test, 30))

        self.assertIsInstance(self.X_train, np.ndarray)
        self.assertIsInstance(self.X_test, np.ndarray)


if __name__ == '__main__':
    unittest.main()
