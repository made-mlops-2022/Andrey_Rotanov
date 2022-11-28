import os
from datetime import timedelta
from airflow.utils.email import send_email_smtp

DATA_DIR = os.getenv('DATA_DIR')
MLRUNS_DIR = os.getenv('MLRUNS_DIR')


def wait_for_file(file_name):
    return os.path.exists(file_name)


def error_callback(message):
    dag_run = message.get('dag_run')
    send_email_smtp(to=default_args['email'], subject=f'DAG {dag_run} has failed')


default_args = {
    'owner': 'Andrey Rotanov',
    'email': ['rotanov07@mail.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': error_callback
}
