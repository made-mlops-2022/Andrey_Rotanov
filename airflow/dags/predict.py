from datetime import datetime
from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from utils import MLRUNS_DIR, DATA_DIR, default_args, wait_for_file

with DAG(
        'predict',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=datetime(2022, 11, 28)
) as dag:
    wait_data = PythonSensor(
        task_id='wait-for-predict-data',
        python_callable=wait_for_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    preprocess = DockerOperator(
        image='airflow-preprocess',
        command='--path-raw-file /data/raw/{{ ds }}/data.csv'
                '--path-target-file /data/raw/{{ ds }}/target.csv'
                '--output-dir /data/processed/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-predict-preprocess',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind')]
    )

    predict = DockerOperator(
        image='airflow-predict',
        command='--input-dir /data/processed/{{ ds }} '
                '--output-dir /data/predictions/{{ ds }}',
        network_mode='host',
        task_id='docker-airflow-predict',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_DIR, target='/data', type='bind'),
                Mount(source=MLRUNS_DIR, target='/mlruns', type='bind')]
    )

    wait_data >> preprocess >> predict
