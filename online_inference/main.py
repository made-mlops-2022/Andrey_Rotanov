import os
import pickle
import pandas as pd
from fastapi import FastAPI
from query_schemes import DataHeartDisease
import requests

app = FastAPI()

model = None
transformer = None


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


@app.on_event("startup")
def load_model():
    path_to_model = os.getenv("PATH_TO_MODEL", 'D:\MADE\MLOps\homework\online_inference\model.pkl')
    path_to_transformer = os.getenv("PATH_TO_TRANSFORMER", 'D:\MADE\MLOps\homework\online_inference\\transformer.pkl')

    if os.getenv("MODEL_DOWNLOAD_FROM_GOOGLE_DISK", 1) is not None:
        if os.getenv("MODEL_FILE_ID_ON_GOOGLE_DISK", 1) is None:
            raise 'Model file id not passed'

        download_file_from_google_drive(os.getenv("MODEL_FILE_ID_ON_GOOGLE_DISK", '1BRLSyq4ODKJSLNaVhKkj-k6jFVh8-i46'), path_to_model)

    if os.getenv("TRANSFORMER_DOWNLOAD_FROM_GOOGLE_DISK", 1) is not None:
        if os.getenv("TRANSFORMER_FILE_ID_ON_GOOGLE_DISK", 1) is None:
            raise 'Transformer file id not passed'

        download_file_from_google_drive(os.getenv("TRANSFORMER_FILE_ID_ON_GOOGLE_DISK", '1QVQ7eJ-yiclsvmom7R6oEP0HheXea2DZ'), path_to_transformer)

    with open(path_to_transformer, 'rb') as file:
        global transformer
        transformer = pickle.load(file)

    with open(path_to_model, 'rb') as file:
        global model
        model = pickle.load(file)


@app.post("/predict")
async def predict(data: DataHeartDisease):
    pd_format = pd.DataFrame([data.dict()])
    pd_format = transformer.transform(pd_format)
    predicted_label = model.predict(pd_format)
    if predicted_label[0]:
        label = 'healthy'
    else:
        label = 'sick'
    return {"predict": label}


@app.get("/health")
async def health():
    if model is None or transformer is None:
        return 'Model is not ready'

    return 'Model is ready'
