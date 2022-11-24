from typing import Literal
from pydantic import BaseModel, validator


class DataHeartDisease(BaseModel):
    age: int
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: int
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @validator('age')
    def check_age(cls, cur_age):
        if cur_age < 0 or cur_age > 105:
            raise ValueError("Error in the age value!")
        return cur_age

    @validator('trestbps')
    def check_trestbps(cls, cur_trestbps):
        if cur_trestbps < 60 or cur_trestbps > 400:
            raise ValueError("Error in the trestbps value!")
        return cur_trestbps

    @validator('chol')
    def check_chol(cls, cur_chol):
        if cur_chol < 0 or cur_chol > 500:
            raise ValueError("Error in the chol value!")
        return cur_chol

    @validator('thalach')
    def check_thalach(cls, cur_thalach):
        if cur_thalach < 50 or cur_thalach > 250:
            raise ValueError("Error in the thalach value!")
        return cur_thalach

    @validator('oldpeak')
    def check_oldpeak(cls, cur_oldpeak):
        if cur_oldpeak < 0.0 or cur_oldpeak > 10:
            raise ValueError("Error in the cur_oldpeak value!")
        return cur_oldpeak