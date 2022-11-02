import pickle
import os
import sys
import unittest
from sklearn.neighbors import KNeighborsClassifier

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'ml_project'
)
sys.path.append(SOURCE_PATH)

import numpy as np
from models.model_fit_predict import (
    train_model,
    create_inference_pipeline,
    predict_model,
    evaluate_model,
    serialize_model)
from data.make_dataset import (
    read_data,
    generate_synthetic_data,
    split_train_val_data)
from features.build_features import (
    extract_target,
    create_transformer,
    make_features)

from enities.train_pipeline_params import read_training_pipeline_params


class TestFitPredictModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config_path = './tests/configs/test_configs.yaml'
        params = read_training_pipeline_params(config_path)
        self.size_generate_data = (540, 14)
        self.size_test = self.size_generate_data[0] * params.splitting_params.val_size
        self.data_gen = generate_synthetic_data(self.size_generate_data[0])

        X, y = extract_target(self.data_gen, params.feature_params.target_col)
        X_train, X_test, y_train, self.y_test = split_train_val_data(X, y, params.splitting_params)
        transformer = create_transformer(params.feature_params)
        transformer.fit(X_train)
        self.model = train_model(X_train, y_train, params.train_params)

        self.y_pred = predict_model(self.model, X_test)

    def test_serialize_model(self):
        pth = 'model.pkl'
        serialize_model(self.model, pth)
        self.assertTrue(os.path.exists(pth))
        with open(pth, 'rb') as f:
            model = pickle.load(f)

        self.assertIsInstance(model, KNeighborsClassifier)
        os.remove(pth)

    def test_predict_model(self):
        self.assertIsInstance(self.y_pred, np.ndarray)
        self.assertEqual(self.y_pred.shape, (self.size_test,))
        self.assertEqual(list(np.unique(self.y_pred)), [0, 1])

    def test_evaluate_model(self):
        metrics = evaluate_model(self.y_pred, self.y_test)
        self.assertIsInstance(metrics, dict)
        self.assertGreaterEqual(metrics['recall'], 0)
