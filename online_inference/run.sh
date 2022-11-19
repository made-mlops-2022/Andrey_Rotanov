#!/bin/bash
docker build -f Dockerfile \
  -t fast_api_model:last .
docker run --env-file .env.example -p 8000:8000 fast_api_model:last