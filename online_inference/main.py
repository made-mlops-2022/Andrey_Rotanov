import os
import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from query_schemes import DataHeartDisease
import requests
import uvicorn

app = FastAPI()

model = None
transformer = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.on_event("startup")
def load_model():
    path_to_model = os.getenv("PATH_TO_MODEL", '/home/andrey/Work/MADE/Andrey_Rotanov/online_inference/model.pkl')
    path_to_transformer = os.getenv("PATH_TO_TRANSFORMER",
                                    '/home/andrey/Work/MADE/Andrey_Rotanov/online_inference/transform.pkl')

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
