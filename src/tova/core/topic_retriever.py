"""Topic-aware RAG retriever for the TOVA chat assistant.

Given a user query and a trained topic model, this module:
1. Embeds topic descriptions with a sentence-transformer
2. Ranks topics by cosine similarity to the query
3. Returns the top-K topics with their top-N representative documents
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))

# Lazy-loaded sentence transformer (heavy import, loaded once)
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded SentenceTransformer all-MiniLM-L6-v2")
    return _encoder


# ---------------------------------------------------------------------------
# TopicRetriever
# ---------------------------------------------------------------------------

class TopicRetriever:
    """Retrieves the most relevant topics and documents for a user query."""

    def __init__(self, model_id: str, model_path: Path, corpus_id: str):
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.corpus_id = corpus_id
        self._tm_folder = self.model_path / "TMmodel"

        self._labels: List[str] = []
        self._descriptions: List[str] = []
        self._summaries: List[str] = []
        self._representative_docs: List[List[Tuple[str, float]]] = []
        self._doc_texts: Dict[str, str] = {}
        self._topic_embeddings: Optional[np.ndarray] = None

        self._load_topic_metadata()
        self._load_corpus_texts()
        self._embed_topics()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _read_lines(self, filename: str) -> List[str]:
        path = self._tm_folder / filename
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    def _load_topic_metadata(self) -> None:
        self._labels = self._read_lines("tpc_labels.txt")
        self._descriptions = self._read_lines("tpc_descriptions.txt")
        self._summaries = self._read_lines("tpc_summaries.txt")

        # Load most representative docs per topic
        jsonl_path = self._tm_folder / "most_representative_docs.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    docs = sorted(
                        entry.get("docs", []),
                        key=lambda d: d["prob"],
                        reverse=True,
                    )
                    self._representative_docs.append(
                        [(d["doc_id"], d["prob"]) for d in docs]
                    )

        n_topics = max(
            len(self._labels),
            len(self._descriptions),
            len(self._representative_docs),
        )
        # Pad lists to the same length
        while len(self._labels) < n_topics:
            self._labels.append(f"Topic {len(self._labels)}")
        while len(self._descriptions) < n_topics:
            self._descriptions.append("")
        while len(self._summaries) < n_topics:
            self._summaries.append("")
        while len(self._representative_docs) < n_topics:
            self._representative_docs.append([])

        logger.info(
            "Loaded metadata for %d topics from model %s",
            n_topics, self.model_id,
        )

    def _load_corpus_texts(self) -> None:
        corpus_dir = DRAFTS_SAVE / self.corpus_id
        data_file = corpus_dir / "data.json"
        if not data_file.exists():
            logger.warning("Corpus data not found at %s", data_file)
            return

        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = raw if isinstance(raw, list) else raw.get("documents", raw)
        for d in items:
            doc_id = str(d.get("id", ""))
            text = d.get("text", d.get("raw_text", ""))
            if doc_id and text:
                self._doc_texts[doc_id] = str(text)

        logger.info(
            "Loaded %d document texts from corpus %s",
            len(self._doc_texts), self.corpus_id,
        )

    def _embed_topics(self) -> None:
        if not self._descriptions:
            return
        # Build rich text representation for each topic
        texts = []
        for i in range(len(self._descriptions)):
            parts = []
            if self._labels[i]:
                parts.append(self._labels[i])
            if self._descriptions[i]:
                parts.append(self._descriptions[i])
            if self._summaries[i]:
                parts.append(self._summaries[i])
            texts.append(". ".join(parts) if parts else f"Topic {i}")

        encoder = _get_encoder()
        self._topic_embeddings = encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        logger.info("Embedded %d topic descriptions", len(texts))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    SIMILARITY_THRESHOLD = 0.35

    @property
    def topic_labels(self) -> List[str]:
        """All topic labels in this model (useful for name verification)."""
        return list(self._labels)

    def retrieve(
        self,
        query: str,
        top_k_topics: int = 3,
        top_n_docs: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-K topics most similar to *query* with top-N docs each.

        Topics whose similarity to the query falls below
        ``SIMILARITY_THRESHOLD`` are excluded so the LLM is never fed
        irrelevant context.

        Returns a list of dicts, one per topic, sorted by descending similarity.
        An empty list signals that the query does not match any known topic.
        """
        if self._topic_embeddings is None or len(self._topic_embeddings) == 0:
            return []

        encoder = _get_encoder()
        query_emb = encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
        )

        similarities = (self._topic_embeddings @ query_emb.T).flatten()
        top_k = min(top_k_topics, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        logger.info(
            "RAG similarity scores (top %d): %s",
            top_k,
            ", ".join(
                f"{self._labels[int(i)]}={float(similarities[int(i)]):.3f}"
                for i in top_indices
            ),
        )

        results = []
        for idx in top_indices:
            idx = int(idx)
            sim = float(similarities[idx])
            if sim < self.SIMILARITY_THRESHOLD:
                continue

            docs_out = []
            for doc_id, prob in self._representative_docs[idx][:top_n_docs]:
                text = self._doc_texts.get(doc_id, "")
                docs_out.append({
                    "doc_id": doc_id,
                    "text": text,
                    "score": float(prob),
                })

            results.append({
                "topic_id": idx,
                "label": self._labels[idx],
                "keywords": self._descriptions[idx],
                "summary": self._summaries[idx],
                "similarity": sim,
                "documents": docs_out,
            })

        if not results:
            logger.info(
                "No topics above threshold %.2f for query: %s",
                self.SIMILARITY_THRESHOLD, query[:120],
            )

        return results


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_retrievers: Dict[str, TopicRetriever] = {}


def get_retriever(model_id: str) -> TopicRetriever:
    """Return a cached TopicRetriever for *model_id*, creating one if needed."""
    if model_id in _retrievers:
        return _retrievers[model_id]

    model_dir = DRAFTS_SAVE / model_id
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    corpus_id = meta.get("corpus_id")
    if not corpus_id:
        raise ValueError(f"Model {model_id} has no corpus_id in metadata")

    retriever = TopicRetriever(
        model_id=model_id,
        model_path=model_dir,
        corpus_id=corpus_id,
    )
    _retrievers[model_id] = retriever
    return retriever


def clear_retriever(model_id: str) -> None:
    _retrievers.pop(model_id, None)
