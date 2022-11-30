FROM python:3.9-slim

COPY split.py requirements.txt /

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "split.py"]