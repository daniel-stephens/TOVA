"""Smoke-test runner for the `tova` CLI (train/infer/model_query).

Usage:
  python tests/test_cli.py
"""

from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

CMD_COLOR = "\033[31m"  # red
RESET_COLOR = "\033[0m"


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data_test" / "bills_sample_100.csv"
CONFIG_PATH = REPO_ROOT / "static" / "config" / "config.yaml"
OUTPUT_ROOT = REPO_ROOT / "data" / "models" / "cli_train_checks"
QUERY_MODEL = "tomotopyLDA"

TRAIN_CASES: Dict[str, Dict] = {
  "tomotopyLDA": {
    "tr_params": {"num_topics": 10, "num_iters": 50}
  },
  "CTM": {
    "tr_params": {"num_topics": 8, "preprocess_text": True}
  },
  "topicGPT": {
    "tr_params": {"sample": 0.05, "temperature": 0.2}
  },
  "OpenTopicRAGModel": {
    "tr_params": {"run_from_web": False, "nr_iterations": 1}
  },
}

INFER_CASES = [
  {
    "model": "tomotopyLDA",
    "text_col": "tokenized_text",
  }
]


def _assert_prerequisites() -> None:
  if not DATA_PATH.exists():
    raise SystemExit(f"Sample data not found at {DATA_PATH}")
  if not CONFIG_PATH.exists():
    raise SystemExit(f"Config file not found at {CONFIG_PATH}")


def _get_sample_doc_ids(count: int = 2) -> List[str]:
  with DATA_PATH.open(newline="", encoding="utf8") as handle:
    reader = csv.DictReader(handle)
    doc_ids: List[str] = []
    for row in reader:
      doc_ids.append(row["id"])
      if len(doc_ids) >= count:
        break
  if not doc_ids:
    raise RuntimeError("Unable to extract doc IDs from sample data")
  return doc_ids


def _run_command(cmd: List[str], label: str) -> None:
  cmd_str = shlex.join(cmd)
  print(f"\n[ CLI ] {label}")
  print(f"{CMD_COLOR}{cmd_str}{RESET_COLOR}")

  result = subprocess.run(cmd, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"{label} failed with exit code {result.returncode}.")


def _run_train_case(model: str, case: Dict) -> Path:
  output_dir = OUTPUT_ROOT / model
  if output_dir.exists():
    shutil.rmtree(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  cmd = [
    sys.executable,
    "-m",
    "src.tova.cli.main",
    "train",
    "run",
    "--model",
    model,
    "--data",
    str(DATA_PATH),
    "--text-col",
    case.get("text_col", "tokenized_text"),
    "--output",
    str(output_dir),
    "--config",
    str(CONFIG_PATH),
  ]

  tr_params = case.get("tr_params")
  if tr_params:
    cmd.extend(["--tr-params", json.dumps(tr_params)])

  _run_command(cmd, f"Training model '{model}'")
  return output_dir


def _run_infer_case(model: str, model_path: Path, text_col: str) -> None:
  cmd = [
    sys.executable,
    "-m",
    "src.tova.cli.main",
    "infer",
    "run",
    "--model-path",
    str(model_path),
    "--data",
    str(DATA_PATH),
    "--text-col",
    text_col,
    "--config",
    str(CONFIG_PATH),
  ]
  _run_command(cmd, f"Inference for '{model}'")


def _run_model_queries(model_path: Path, doc_ids: List[str]) -> None:
  docs_arg = ",".join(doc_ids)
  base = [sys.executable, "-m", "src.tova.cli.main", "model_query"]
  commands = [
    (
      "Model info query",
      base
      + [
        "model-info",
        "--model-path",
        str(model_path),
        "--config",
        str(CONFIG_PATH),
      ],
    ),
    (
      "Topic info query",
      base
      + [
        "topic-info",
        "--topic-id",
        "0",
        "--model-path",
        str(model_path),
        "--config",
        str(CONFIG_PATH),
      ],
    ),
    (
      "Thetas by doc IDs",
      base
      + [
        "get-thetas-docs-by-id",
        "--docs-ids",
        docs_arg,
        "--model-path",
        str(model_path),
        "--config",
        str(CONFIG_PATH),
      ],
    ),
  ]

  for label, cmd in commands:
    _run_command(cmd, label)


def main() -> int:
  _assert_prerequisites()
  OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

  failures = []
  trained_paths: Dict[str, Path] = {}

  for model, case in TRAIN_CASES.items():
    try:
      trained_paths[model] = _run_train_case(model, case)
    except Exception as exc:  # pylint: disable=broad-except
      failures.append((f"train:{model}", str(exc)))

  for infer_case in INFER_CASES:
    model = infer_case["model"]
    model_path = trained_paths.get(model)
    if not model_path:
      failures.append((f"infer:{model}", "Model not trained"))
      continue
    try:
      _run_infer_case(model, model_path, infer_case.get("text_col", "tokenized_text"))
    except Exception as exc:  # pylint: disable=broad-except
      failures.append((f"infer:{model}", str(exc)))

  query_model_path = trained_paths.get(QUERY_MODEL)
  if query_model_path:
    try:
      doc_ids = _get_sample_doc_ids()
      _run_model_queries(query_model_path, doc_ids)
    except Exception as exc:  # pylint: disable=broad-except
      failures.append(("model_query", str(exc)))
  else:
    failures.append(("model_query", f"Query model '{QUERY_MODEL}' was not trained"))

  if failures:
    print("\nThe following CLI checks failed:")
    for scope, msg in failures:
      print(f" - {scope}: {msg}")
    return 1

  print("\nAll CLI commands completed successfully.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
