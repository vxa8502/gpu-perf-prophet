#!/usr/bin/env sh
# Starts both processes that make up the deployed app: uvicorn (FastAPI) in the background, and Streamlit (the UI HF Spaces exposes on $PORT) in the foreground as the container's main process. Streamlit calls the API over HTTP (app/api_client.py) instead of importing the predictor/recommender in-process, so every UI interaction is real API traffic.
set -e

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

exec streamlit run app/streamlit_app.py \
    --server.port="${PORT:-7860}" --server.address=0.0.0.0 \
    --server.headless=true --browser.gatherUsageStats=false
