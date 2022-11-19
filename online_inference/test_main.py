import json
# import unittest
from fastapi.testclient import TestClient
import pytest
from main import app, load_model

client = TestClient(app)

@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()

class RequestParameter():
    parameter: str
    value: float or int
    status_code: int

    def __init__(self, parameter, value, status_code):
        self.parameter = parameter
        self.value = value
        self.status_code = status_code


# class TestApi(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super(TestApi, self).__init__(*args, **kwargs)
#         # self.client = TestClient(app)

def test_predict():
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
    data = json.dumps(request_data)
    response = client.post('/predict',
                                json=data)
    # self.assertEqual(response.status_code, 200)
    # self.assertEqual(response.status_code, 200)

def test_health():
    response = client.get('/health')
    # self.assertEqual(response.status_code, 200)
    # self.assertEqual(response.json(), 'Model is ready')

def test_numerical_fields_case_age():
    request_template = {
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
    response = client.post(
        '/predict',
        json = json.dumps(request_template)
    )
    # self.assertEqual(response.status_code, 200)

    parameters_replace = []
    parameters_replace.append(RequestParameter('age', -1, 422))
    for cur_replace in parameters_replace:
        request_data = request_template.copy()
        request_data[cur_replace.parameter] = cur_replace.value
        response = client.post('/predict/', json=json.dumps(request_data))
        # self.assertEqual(response.status_code, cur_replace.status_code)

def test_categorial_fields():
    pass

# if __name__ == '__main__':
#     unittest.main()
