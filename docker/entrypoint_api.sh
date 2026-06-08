#!/bin/sh
set -e

CONFIG=/app/static/config/config.yaml

# Read embedding model from config; fall back to empty string if not set or preload disabled
PRELOAD=$(python -c "
import sys, yaml
try:
    cfg = yaml.safe_load(open('$CONFIG'))
    tm = cfg.get('topic_modeling', {})
    otr = tm.get('opentopicrag') or tm.get('OpenTopicRAG') or {}
    print(otr.get('preload_embedding_on_startup', False))
except Exception:
    print(False)
")

if [ "$PRELOAD" = "True" ]; then
    MODEL=$(python -c "
import sys, yaml
try:
    cfg = yaml.safe_load(open('$CONFIG'))
    tm = cfg.get('topic_modeling', {})
    otr = tm.get('opentopicrag') or tm.get('OpenTopicRAG') or {}
    print(otr.get('embedding_model', ''))
except Exception:
    print('')
")
    if [ -n "$MODEL" ]; then
        echo "[entrypoint] Downloading embedding model '$MODEL' before startup..."
        python -c "
from transformers import AutoTokenizer, AutoModel
model = '$MODEL'
AutoTokenizer.from_pretrained(model, trust_remote_code=True)
AutoModel.from_pretrained(model, trust_remote_code=True)
print('[entrypoint] Model ready.')
"
    fi
fi

exec python -m uvicorn tova.api.main:app --host 0.0.0.0 --port 11000
