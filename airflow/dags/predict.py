from datetime import datetime
from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from utils import LOCAL_MLRUNS_DIR, LOCAL_DATA_DIR, default_args, wait_for_file

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
        command='--input-dir /data/raw/{{ ds }}/data.csv '
                '--test-size 0.2 '
                '--path-target-file /data/raw/{{ ds }}/target.csv '
                '--output-dir /data/processed/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-predict-preprocess',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    predict = DockerOperator(
        image='airflow-predict',
        command='--path-to-data /data/processed/{{ ds }} '
                '--output-dir /data/predictions/{{ ds }}',
        network_mode='host',
        task_id='docker-airflow-predict',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind'),
                Mount(source=LOCAL_MLRUNS_DIR, target='/mlruns', type='bind')]
    )

    wait_data >> preprocess >> predict
