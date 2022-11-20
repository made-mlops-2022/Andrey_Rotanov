import json
import pandas as pd
import requests
import logging

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
stream_handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

PATH_DATASET_FILE = 'script_for_queries/synthetic_data.csv'
SERVER_ADDRESS = 'http://localhost:8000/predict'
TARGET_COLUMN = 'condition'

if __name__ == "__main__":
    logger.info(f'Dataset path: {PATH_DATASET_FILE}')
    logger.info(f'Server address: {SERVER_ADDRESS}')

    data = pd.read_csv(PATH_DATASET_FILE)
    data, target = data.drop(TARGET_COLUMN, axis=1).to_dict(orient='records'), data[TARGET_COLUMN]
    for cur_request in data:
        response = requests.post(SERVER_ADDRESS, json.dumps(cur_request))
        logger.info(f'Response Body: {response}')
