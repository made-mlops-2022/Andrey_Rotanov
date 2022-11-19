#!/bin/bash
docker build -f Dockerfile -t fast_api_model .
docker run --env-file .env -p 8000:8000 fast_api_model