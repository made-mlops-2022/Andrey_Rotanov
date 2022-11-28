FROM python:3.9-slim

COPY validate.py requirements.txt /

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "validate.py"]