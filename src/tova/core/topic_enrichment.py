"""Generate labels and summaries for an existing model without retraining it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil

import pandas as pd

from tova.topic_models.tm_model import TMmodel
from tova.utils.common import load_yaml_config_file, write_json_atomic


def enrich_model(model_id: str, *, drafts_root: Path | None = None,
                 config_path: str = "static/config/config.yaml",
                 provider: str | None = None, model: str | None = None,
                 server: str | None = None) -> dict:
    root = Path(drafts_root or os.getenv("DRAFTS_SAVE", "/data/drafts")).resolve()
    model_dir = (root / model_id).resolve()
    if model_dir.parent != root or not model_dir.is_dir():
        raise ValueError("Model ID must identify a model in the draft store.")
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    corpus_dir = (root / metadata["corpus_id"]).resolve()
    if corpus_dir.parent != root:
        raise ValueError("Invalid corpus ID.")
    records = json.loads((corpus_dir / "data.json").read_text())
    records = records if isinstance(records, list) else records["documents"]
    by_id = {str(row["id"]): row for row in records}
    folder = model_dir / "TMmodel"
    # The saved theta representation preserves the training row order. Align
    # by ID so reordering the corpus cannot silently change the prompt evidence.
    order = list(json.loads((folder / "thetas_representation.json").read_text()))
    if not order or any(doc_id not in by_id for doc_id in order):
        raise ValueError("Training document IDs cannot be matched to the corpus.")
    frame = pd.DataFrame([by_id[doc_id] for doc_id in order])
    logger = logging.getLogger(__name__)
    config = load_yaml_config_file(config_path, "topic_modeling", logger)
    general, traditional = config.get("general", {}), config.get("traditional", {})
    provider = provider or general.get("llm_provider", "ollama")
    model = model or general.get("llm_model_type")
    server = server or general.get("llm_server")
    tm = TMmodel(
        folder, df_corpus_train=frame, config_path=config_path, logger=logger,
        llm_provider=provider, llm_model_type=model, llm_server=server,
        labeller_prompt=traditional.get("labeller_prompt", "src/tova/prompter/prompts/labelling_dft.txt"),
        summarizer_prompt=traditional.get("summarizer_prompt", "src/tova/prompter/prompts/summarization_dft.txt"),
    )
    tm._load_thetas()
    if tm._thetas.shape[0] != len(frame):
        raise ValueError("Saved document matrix and corpus have different lengths.")
    # Generate everything before changing the served artifacts. A backend
    # failure must leave the previous labels and summaries intact.
    labels = [" ".join(text.split()).strip('"') for _, text in tm.get_tpc_labels()]
    summaries = [" ".join(text.split()) for _, text in tm.get_tpc_summaries()]
    if len(labels) != tm._ntopics or len(summaries) != tm._ntopics:
        raise ValueError("The LLM did not return an output for every topic.")
    if any(not label or re.fullmatch(r"Topic\s+\d+", label, re.I) for label in labels):
        raise ValueError("The LLM returned an empty or placeholder topic label.")
    if any(not summary or summary.startswith("Placeholder for summary") for summary in summaries):
        raise ValueError("The LLM returned an empty or placeholder topic summary.")
    generated_at = datetime.now(timezone.utc)
    backup = model_dir / "enrichment-backups" / generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    backup.mkdir(parents=True)
    shutil.copy2(metadata_path, backup / "metadata.json")
    for name in ("tpc_labels.txt", "tpc_summaries.txt"):
        if (folder / name).exists():
            shutil.copy2(folder / name, backup / name)
    for name, values in (("tpc_labels.txt", labels), ("tpc_summaries.txt", summaries)):
        temporary = folder / (name + ".tmp")
        temporary.write_text("\n".join(values) + "\n", encoding="utf-8")
        temporary.replace(folder / name)
    result = {"provider": provider, "model": model, "server": server,
              "generated_at": generated_at.isoformat(), "topics": len(labels),
              "backup": str(backup)}
    metadata["topic_enrichment"] = result
    write_json_atomic(metadata_path, metadata)
    return {**result, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--config", default="static/config/config.yaml")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--server")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = enrich_model(args.model_id, config_path=args.config, provider=args.provider,
                          model=args.model, server=args.server)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
