FROM python:3.9-slim

WORKDIR /app/src

COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY main.py query_schemes.py run_app.sh /app/src/
CMD ["bash", "run_app.sh"]