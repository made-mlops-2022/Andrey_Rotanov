import unittest
from fastapi.testclient import TestClient
from main import app, load_model

client = TestClient(app)


class RequestParameter():
    parameter: str
    value: float or int
    status_code: int

    def __init__(self, parameter, value, status_code):
        self.parameter = parameter
        self.value = value
        self.status_code = status_code


class TestApi(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestApi, self).__init__(*args, **kwargs)
        self.client = TestClient(app)
        load_model()
        self.request_template = {
            'age': 5,
            'sex': 1,
            'cp': 3,
            'trestbps': 155,
            'chol': 165,
            'fbs': 1,
            'restecg': 0,
            'thalach': 91,
            'exang': 0,
            'oldpeak': 1.9,
            'slope': 0,
            'ca': 0,
            'thal': 2
        }

    def test_predict(self):
        request_data = {
            'age': 5,
            'sex': 1,
            'cp': 3,
            'trestbps': 155,
            'chol': 165,
            'fbs': 1,
            'restecg': 0,
            'thalach': 91,
            'exang': 0,
            'oldpeak': 1.7,
            'slope': 0,
            'ca': 0,
            'thal': 2
        }
        response = self.client.post('/predict',
                                    json=request_data)
        self.assertEqual(response.status_code, 200)

    def test_health(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), 'Model is ready')

    def test_numerical_fields_case_age(self):
        response = client.post(
            '/predict',
            json=self.request_template
        )
        self.assertEqual(response.status_code, 200)

        parameters_replace = []
        parameters_replace.append(RequestParameter('age', -1, 422))
        parameters_replace.append(RequestParameter('age', 55, 200))
        parameters_replace.append(RequestParameter('age', 106, 422))

        parameters_replace.append(RequestParameter('trestbps', 40, 422))
        parameters_replace.append(RequestParameter('trestbps', 100, 200))
        parameters_replace.append(RequestParameter('trestbps', 201, 422))

        parameters_replace.append(RequestParameter('chol', -1, 422))
        parameters_replace.append(RequestParameter('chol', 100, 200))
        parameters_replace.append(RequestParameter('chol', 501, 422))

        parameters_replace.append(RequestParameter('thalach', 48, 422))
        parameters_replace.append(RequestParameter('thalach', 100, 200))
        parameters_replace.append(RequestParameter('thalach', 251, 422))

        parameters_replace.append(RequestParameter('oldpeak', -1, 422))
        parameters_replace.append(RequestParameter('oldpeak', 1.9, 200))
        parameters_replace.append(RequestParameter('oldpeak', 10.1, 422))
        for cur_replace in parameters_replace:
            request_data = self.request_template.copy()
            request_data[cur_replace.parameter] = cur_replace.value
            response = client.post('/predict/', json=request_data)
            self.assertEqual(response.status_code, cur_replace.status_code)

    def test_categorial_fields(self):
        response = client.post(
            '/predict',
            json=self.request_template
        )
        self.assertEqual(response.status_code, 200)

        parameters_replace = []
        parameters_replace.append(RequestParameter('sex', -1, 422))
        parameters_replace.append(RequestParameter('sex', 1, 200))
        parameters_replace.append(RequestParameter('sex', 2, 422))

        parameters_replace.append(RequestParameter('cp', -1, 422))
        parameters_replace.append(RequestParameter('cp', 2, 200))
        parameters_replace.append(RequestParameter('cp', 4, 422))

        parameters_replace.append(RequestParameter('fbs', -1, 422))
        parameters_replace.append(RequestParameter('fbs', 0, 200))
        parameters_replace.append(RequestParameter('fbs', 2, 422))

        parameters_replace.append(RequestParameter('restecg', -1, 422))
        parameters_replace.append(RequestParameter('restecg', 1, 200))
        parameters_replace.append(RequestParameter('restecg', 3, 422))

        parameters_replace.append(RequestParameter('exang', -1, 422))
        parameters_replace.append(RequestParameter('exang', 0, 200))
        parameters_replace.append(RequestParameter('exang', 2, 422))

        parameters_replace.append(RequestParameter('slope', -1, 422))
        parameters_replace.append(RequestParameter('slope', 0, 200))
        parameters_replace.append(RequestParameter('slope', 3, 422))

        parameters_replace.append(RequestParameter('ca', -1, 422))
        parameters_replace.append(RequestParameter('ca', 0, 200))
        parameters_replace.append(RequestParameter('ca', 4, 422))

        parameters_replace.append(RequestParameter('thal', -1, 422))
        parameters_replace.append(RequestParameter('thal', 0, 200))
        parameters_replace.append(RequestParameter('thal', 3, 422))
        for cur_replace in parameters_replace:
            request_data = self.request_template.copy()
            request_data[cur_replace.parameter] = cur_replace.value
            response = client.post('/predict/', json=request_data)
            self.assertEqual(response.status_code, cur_replace.status_code)


if __name__ == '__main__':
    unittest.main()
