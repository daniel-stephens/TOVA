import json
from pathlib import Path

EXAMPLES_DIR = Path("static/examples")

TRAIN_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "train_request.json").read_text(encoding="utf-8"))

INFER_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "infer_request.json").read_text(encoding="utf-8"))

MODEL_INFO_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "model_info_request.json").read_text(encoding="utf-8"))

TOPIC_INFO_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "topic_info_request.json").read_text(encoding="utf-8"))

THETAS_BY_DOCS_IDS_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "thetas_by_docs_id.json").read_text(encoding="utf-8"))