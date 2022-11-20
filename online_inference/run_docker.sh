#!/bin/bash
docker build -f Dockerfile -t fast_api_model .
docker tag fast_api_model:latest andrey506/made_mlops_homework:latest
docker run --env-file .env -p 8000:8000 fast_api_model