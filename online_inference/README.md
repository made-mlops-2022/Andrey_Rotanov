## Инструкция по запуску
Перед запуском необходимо создать `.env` файл, скопировать туда содержимое `.env.example` и указать значения соответствующих переменных
Описание переменных:
```
PORT --- Порт, на котором будет запущен FastApi
PATH_TO_MODEL -- Путь, по которому будет сохранена скачанная модель
PATH_TO_TRANSFORMER --- Путь, по которому будет сохранен скачанный трансформер
```

Перед запуском контейнера также необходимо загрузить image docker выполнив следующую команду:
```commandline
docker pull andrey506/made_mlops_homework:latest
```

Для запуска Docker контейнера необходимо выполнить следующую команду:
```commandline
docker run --env-file .env -p 8000:8000 andrey506/made_mlops_homework:v1
```

Для сборки  необходимо выполнить следующую команду:
```commandline
docker build -f Dockerfile -t fast_api_model .
```
ДЛя загрузки образа в `Docker Hub` необходимо авторизироваться в нем и выполнить следующие команды:
```commandline
docker login
docker tag fast_api_model:latest andrey506/made_mlops_homework:latest
docker push andrey506/made_mlops_homework:latest
```

Для запуска FastApi без использования контейнера необходимо выполнить следующие команды:
```commandline
source .env
bash run_app.sh
```

Для запуска тестов необходимо выполнить:
```commandline
bash run_test.sh
```

Для тестирования запросов необходимо выполнить следующую команду:
```commandline
python3 script_for_queries/sending_requests.py 
```

## Оптимизация image
1) Эксперементальным путем на работе было установлено что чем меньше инструкций, тем меньше слоев и соответственно размер
2) Добавил в `.dockerignore` все, кроме копируемых файлов
3) Использовал урезанную версию `python:3.9-slim`