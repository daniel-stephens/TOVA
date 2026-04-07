"""Active learning module for TOVA.

Provides per-model active learning sessions that combine topic model
representations (thetas) with TF-IDF features to recommend the most
informative document for a human to label next.
"""

import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))

# ---------------------------------------------------------------------------
# ActiveLearningSession
# ---------------------------------------------------------------------------

class ActiveLearningSession:
    """Manages an active-learning loop for a single trained topic model.

    On initialisation the session builds a document feature matrix by
    concatenating topic-model thetas (for traditional models) with TF-IDF
    features.  It then accepts human labels, retrains a lightweight
    classifier, and recommends the next document to label based on
    prediction entropy.
    """

    def __init__(self, model_id: str, model_path: Path, corpus_id: str):
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.corpus_id = corpus_id

        self._labels: Dict[str, str] = {}
        self._classifier: Optional[LogisticRegression] = None

        self._doc_ids: List[str] = []
        self._doc_texts: Dict[str, str] = {}
        self._feature_matrix: Optional[sparse.csr_matrix] = None

        self._al_dir = self.model_path / "active_learning"
        self._al_dir.mkdir(parents=True, exist_ok=True)

        self._build_features()
        self._load_state()

        if self._has_enough_labels():
            self._retrain()

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def _is_traditional_model(self) -> bool:
        meta_path = self.model_path / "metadata.json"
        if not meta_path.exists():
            return False
        with open(meta_path, "r") as f:
            meta = json.load(f)
        model_type: str = meta.get("type", "")
        return "traditional" in model_type

    def _load_corpus_documents(self) -> List[Dict[str, Any]]:
        corpus_dir = DRAFTS_SAVE / self.corpus_id
        data_file = corpus_dir / "data.json"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Corpus data not found at {data_file}"
            )
        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        docs: List[Dict[str, Any]] = []
        items = raw if isinstance(raw, list) else raw.get("documents", raw)
        for d in items:
            doc_id = str(d.get("id", ""))
            text = d.get("text", d.get("raw_text", ""))
            if doc_id and text:
                docs.append({"id": doc_id, "text": str(text)})
        return docs

    def _load_thetas(self) -> Optional[sparse.csr_matrix]:
        thetas_path = self.model_path / "TMmodel" / "thetas.npz"
        if not thetas_path.exists():
            return None
        return sparse.load_npz(thetas_path)

    def _build_features(self) -> None:
        docs = self._load_corpus_documents()
        if not docs:
            raise ValueError(f"No documents found for corpus {self.corpus_id}")

        self._doc_ids = [d["id"] for d in docs]
        self._doc_texts = {d["id"]: d["text"] for d in docs}
        texts = [d["text"] for d in docs]

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        is_traditional = self._is_traditional_model()
        if is_traditional:
            thetas = self._load_thetas()
            if thetas is not None:
                # thetas rows are aligned with the training corpus order.
                # If the corpus doc count matches thetas rows, concatenate
                # directly.  Otherwise fall back to TF-IDF only.
                if thetas.shape[0] == tfidf_matrix.shape[0]:
                    thetas_norm = normalize(thetas, axis=1, norm="l1")
                    self._feature_matrix = sparse.hstack(
                        [thetas_norm, tfidf_matrix], format="csr"
                    )
                    logger.info(
                        "Built feature matrix [thetas+tfidf] shape %s for model %s",
                        self._feature_matrix.shape, self.model_id,
                    )
                    return
                else:
                    logger.warning(
                        "Thetas rows (%d) != corpus docs (%d); using TF-IDF only",
                        thetas.shape[0], tfidf_matrix.shape[0],
                    )

        self._feature_matrix = tfidf_matrix
        logger.info(
            "Built feature matrix [tfidf-only] shape %s for model %s",
            self._feature_matrix.shape, self.model_id,
        )

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def add_label(self, doc_id: str, label: str) -> None:
        if doc_id not in self._doc_texts:
            raise KeyError(f"Document {doc_id} not found in corpus")
        self._labels[doc_id] = label
        self._save_state()
        if self._has_enough_labels():
            self._retrain()

    def remove_label(self, doc_id: str) -> None:
        if doc_id not in self._labels:
            raise KeyError(f"No label for document {doc_id}")
        del self._labels[doc_id]
        self._save_state()
        if self._has_enough_labels():
            self._retrain()
        else:
            self._classifier = None

    def get_labels(self) -> Dict[str, str]:
        return dict(self._labels)

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------

    def _has_enough_labels(self) -> bool:
        """Need >= 2 distinct classes with >= 1 sample each."""
        return len(set(self._labels.values())) >= 2

    def _retrain(self) -> None:
        labeled_ids = list(self._labels.keys())
        idx_map = {did: i for i, did in enumerate(self._doc_ids)}
        indices = [idx_map[did] for did in labeled_ids]
        X = self._feature_matrix[indices]
        y = [self._labels[did] for did in labeled_ids]

        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs", multi_class="multinomial"
        )
        clf.fit(X, y)
        self._classifier = clf
        logger.info(
            "Retrained classifier for model %s on %d labels (%d classes)",
            self.model_id, len(y), len(set(y)),
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(self) -> Tuple[str, str, bool]:
        """Return (doc_id, text, classifier_driven).

        If no classifier is trained yet, return a random unlabeled document.
        Otherwise return the unlabeled document with the highest prediction
        entropy.
        """
        unlabeled = [d for d in self._doc_ids if d not in self._labels]
        if not unlabeled:
            raise ValueError("All documents have been labeled")

        if self._classifier is None:
            chosen = random.choice(unlabeled)
            return chosen, self._doc_texts[chosen], False

        idx_map = {did: i for i, did in enumerate(self._doc_ids)}
        indices = [idx_map[d] for d in unlabeled]
        X_unlabeled = self._feature_matrix[indices]

        probs = self._classifier.predict_proba(X_unlabeled)
        # Entropy: H = -sum(p * log(p)), with 0*log(0) = 0
        log_probs = np.log(probs + 1e-12)
        entropies = -np.sum(probs * log_probs, axis=1)
        best_local_idx = int(np.argmax(entropies))
        chosen = unlabeled[best_local_idx]
        return chosen, self._doc_texts[chosen], True

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "num_labeled": len(self._labels),
            "num_unlabeled": len(self._doc_ids) - len(self._labels),
            "num_total": len(self._doc_ids),
            "class_distribution": dict(Counter(self._labels.values())),
            "classifier_trained": self._classifier is not None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        labels_path = self._al_dir / "labels.json"
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(self._labels, f, indent=2)

    def _load_state(self) -> None:
        labels_path = self._al_dir / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self._labels = json.load(f)
            logger.info(
                "Loaded %d existing labels for model %s",
                len(self._labels), self.model_id,
            )


# ---------------------------------------------------------------------------
# ActiveLearningManager  (module-level singleton cache)
# ---------------------------------------------------------------------------

_sessions: Dict[str, ActiveLearningSession] = {}


def get_session(model_id: str) -> ActiveLearningSession:
    """Return cached session or create a new one for *model_id*."""
    if model_id in _sessions:
        return _sessions[model_id]

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
        raise ValueError(
            f"Model {model_id} has no corpus_id in metadata; "
            "cannot initialise active learning"
        )

    session = ActiveLearningSession(
        model_id=model_id,
        model_path=model_dir,
        corpus_id=corpus_id,
    )
    _sessions[model_id] = session
    return session


def clear_session(model_id: str) -> None:
    """Remove a cached session (e.g. when a model is deleted)."""
    _sessions.pop(model_id, None)
