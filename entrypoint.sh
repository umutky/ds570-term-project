#!/bin/bash
set -e

MODEL_PATH=/app/outputs/models/lgbm_tweedie.pkl

if [ ! -f "$MODEL_PATH" ]; then
    echo "First run: fetching data and training models. This takes a few minutes..."
    rf-fetch
    rf-process
    rf-train
    rf-predict
fi

exec streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
