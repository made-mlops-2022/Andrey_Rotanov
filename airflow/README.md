# Начало работы
Перед запуском контейнера airflow необходимо экспортировать переменные окружения:
```commandline
export DATA_DIR=$(pwd)/data
export MLRUNS_DIR=$(pwd)/mlruns
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
```

Для запуска контейнера необходимо выполнить следующую команду:
```commandline
docker-compose up --build
```

Для запуска тестов необходимо выполнить следующую команду:
```commandline

```