# Начало работы
Перед запуском контейнера airflow необходимо экспортировать переменные окружения:
Linux:
```commandline
export LOCAL_DATA_DIR=$(pwd)/data
export PASSWORD=YOUR_PASSWORD
export LOCAL_MLRUNS_DIR=$(pwd)/mlruns
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
```
Windows:
```commandline
$env:PASSWORD=YOUR_PASSWORD
$env:LOCAL_DATA_DIR="$(pwd)/data"
$env:LOCAL_MLRUNS_DIR="$(pwd)/mlruns"
$env:FERNET_KEY="$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")"
```

Для запуска контейнера необходимо выполнить следующую команду:
```commandline
docker-compose up --build
```

Для запуска тестов необходимо выполнить следующую команду:
```commandline
docker exec -it airflow_ml_dags_scheduler_1 bash
pip3 install pytest
python3 -m pytest --disable-warnings tests/tests_dag_structure.py
python3 -m pytest --disable-warnings tests/tests_dag_loading.py
```