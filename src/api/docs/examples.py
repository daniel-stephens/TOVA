import json
from pathlib import Path

EXAMPLES_DIR = Path("static/examples")

TRAIN_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "train_request.json").read_text(encoding="utf-8"))
TRAIN_FILE_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "train_file_request.json").read_text(encoding="utf-8"))

INFER_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "infer_request.json").read_text(encoding="utf-8"))
INFER_FILE_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "infer_file_request.json").read_text(encoding="utf-8"))

MODEL_INFO_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "model_info_request.json").read_text(encoding="utf-8"))

TOPIC_INFO_REQUEST_EXAMPLE = json.loads((EXAMPLES_DIR / "topic_info_request.json").read_text(encoding="utf-8"))