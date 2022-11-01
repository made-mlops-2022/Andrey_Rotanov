import unittest
import pandas as pd
import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'ml_project'
)
sys.path.append(SOURCE_PATH)
from data.make_dataset import *
from features.build_features import *
from enities.splitting_params import SplittingParams


class TestData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = ["fbs",
                        "sex",
                        "cp",
                        "restecg",
                        "exang",
                        "thal",
                        "slope",
                        "ca",
                        "chol",
                        "age",
                        "oldpeak",
                        "thalach",
                        "trestbps",
                        'condition']
        self.path_file = 'tests/synthetic_data/synthetic_data.csv'
        self.size_generate_data = (540, 14)
        self.data_gen = generate_synthetic_data(self.size_generate_data[0])
        self.data_gen.to_csv(self.path_file, index=False)

        self.data = read_data(self.path_file)
        self.target_column = 'condition'
        self.test_size = 0.2
        self.sh = self.data.columns
        self.X, self.y = extract_target(self.data, self.target_column)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            split_train_val_data(self.X, self.y,
                                 SplittingParams(
                                     val_size=self.test_size,
                                     random_state=37
                                 ))

    def testing_read_data(self):
        self.assertEqual(self.data.shape, self.size_generate_data)
        self.assertIsInstance(self.data, pd.DataFrame)
        data_columns = self.data.columns
        for cur_col in self.columns:
            self.assertIn(cur_col, data_columns)

    def testing_extract_target(self):
        self.assertIsInstance(self.X, pd.DataFrame)
        self.assertIsInstance(self.y, pd.Series)
        self.assertEqual(self.X.shape, (self.size_generate_data[0], self.size_generate_data[1] - 1))
        self.assertEqual(self.y.shape, (self.size_generate_data[0],))

    def testing_spliting_data(self):
        train_size = int(self.size_generate_data[0] * (1 - self.test_size))
        test_size = self.size_generate_data[0] - train_size
        self.assertIsInstance(self.X_train, pd.DataFrame)
        self.assertIsInstance(self.X_test, pd.DataFrame)
        self.assertIsInstance(self.y_test, pd.Series)
        self.assertIsInstance(self.y_train, pd.Series)

        self.assertEqual(self.X_train.shape, (train_size, 13))
        self.assertEqual(self.y_train.shape, (train_size,))
        self.assertEqual(self.y_test.shape, (test_size,))
        self.assertEqual(self.X_test.shape, (test_size, 13))


if __name__ == '__main__':
    unittest.main()
