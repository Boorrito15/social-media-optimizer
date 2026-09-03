#!/usr/bin/env bash
# Run the predictor API + Streamlit UI for the Social Media Optimizer.
#
#   ./run.sh api     # start FastAPI on :8000 (blocks)
#   ./run.sh app     # start Streamlit UI on :8501 (uses SMO_API default)
#   ./run.sh train   # (re)train models from data/processed/processed.csv
#   ./run.sh all     # start API in background, then Streamlit
set -e
cd "$(dirname "$0")"

case "${1:-}" in
  train)
    .venv/bin/python -W ignore -m src.ml.train
    ;;
  train-keras)
    .venv/bin/python -W ignore -m src.ml.train_keras
    ;;
  api)
    exec .venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
    ;;
  app)
    export SMO_API="${SMO_API:-http://127.0.0.1:8000}"
    exec .venv/bin/streamlit run streamlit_app.py --server.port "${SMO_PORT:-8501}"
    ;;
  all)
    echo "Starting API on :8000 ..."
    .venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > /tmp/smo_api.log 2>&1 &
    API_PID=$!
    trap "kill $API_PID 2>/dev/null || true" EXIT
    echo "Starting Streamlit on :8501 ..."
    export SMO_API="http://127.0.0.1:8000"
    .venv/bin/streamlit run streamlit_app.py --server.port 8501
    ;;
  *)
    echo "usage: $0 {train|api|app|all}"
    exit 1
    ;;
esac
