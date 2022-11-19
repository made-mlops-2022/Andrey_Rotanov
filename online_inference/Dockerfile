#FROM python:3.8.15-slim-buster
FROM python:3.9-slim

WORKDIR /app/src

COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY main.py query_schemes.py  /app/src
CMD uvicorn main:app --workers 1 --host 0.0.0.0 --port 8000