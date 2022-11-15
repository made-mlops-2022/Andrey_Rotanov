# MLOPS 2022
## Rotanov Andrey
# Запуск проекта

Перед запуском проекта необходимо установить все зависимости.
```commandline
pip3 install -r requirements.txt
```

### Train 
Настройка параметров обучения модели производится в конфиге в каталоге `configs/train_config.yaml`
Для тренировки модели необходимо выполнить следующую команду:
```commandline
python3 .\ml_project\train_pipeline.py
```

### Predict
Для вызова функции `predict` необходимо выполнить следующую команду:
```commandline
python3 .\ml_project\predict_pipeline.py --path_to_model <путь до модели> --path_to_transformer <путь до трансформера> --path_to_data <путь до csv файла с данными> --path_to_prediction <путь, результата предсказания> 
```
Пример 
```commandline
python3 .\ml_project\predict_pipeline.py --path_to_model ./models/models/model.pkl --path_to_transformer models/transformers/transform.pkl --path_to_data ./data/raw/heart_cleveland_upload.csv --path_to_prediction ./models/predictions/predict.csv
```

Для прогона тестов необходимо выполнить следующую команду:
```commandline
python3 -m unittest tests
```
