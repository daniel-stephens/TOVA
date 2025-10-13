from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as scipy_entropy


# ---------- I/O helpers ----------
class CorpusNotFoundError(FileNotFoundError): ...
class InvalidCorpusError(ValueError): ...

def _load_corpus_documents(corpus_dir: Path) -> List[Dict[str, Any]]:
    """
    Reads <corpus_dir>/corpus.json and returns normalized documents:
    [{ id: str, raw_text: str }, ...]
    """
    corpus_file = corpus_dir / "corpus.json"
    if not corpus_file.exists():
        raise CorpusNotFoundError(f"corpus.json not found in {corpus_dir}")

    try:
        payload = json.loads(corpus_file.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise InvalidCorpusError(f"Failed to parse corpus.json: {e}")

    docs = payload.get("documents") or []
    if not isinstance(docs, list) or not docs:
        raise InvalidCorpusError("No documents found in corpus.json")

    out: List[Dict[str, Any]] = []
    for d in docs:
        raw = d.get("text") if d.get("text") is not None else d.get("raw_text")
        if raw is None:
            continue
        out.append({"id": str(d.get("id")), "raw_text": str(raw)})
    if not out:
        raise InvalidCorpusError("No valid documents with text found.")
    return out

def _save_analysis_outputs(
    corpus_dir: Path,
    metrics: Dict[str, Any],
    tfidf_details: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Writes analysis files in the corpus directory:
      - analysis_metrics.json
      - tfidf_details.jsonl
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = corpus_dir / "analysis_metrics.json"
    tfidf_path = corpus_dir / "tfidf_details.jsonl"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with tfidf_path.open("w", encoding="utf-8") as f:
        for item in tfidf_details:
            f.write(json.dumps(item))
            f.write("\n")

    return {"metrics": str(metrics_path), "tfidf": str(tfidf_path)}


# ---------- Analytics primitives ----------
# _build_tfidf: catch empty vocabulary
def _build_tfidf(texts: List[str], max_features: int = 1000):
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    try:
        mat = vec.fit_transform(texts)
    except ValueError as e:  # e.g., "empty vocabulary; perhaps the documents only contain stop words"
        raise InvalidCorpusError(str(e))
    if mat.shape[1] == 0:
        raise InvalidCorpusError("TF-IDF produced 0 features (empty vocabulary).")
    return vec, mat


# _compute_lsi: keep n_components <= num features
def _compute_lsi(tfidf_matrix, max_components: int = 200, random_state: int = 42):
    n_features = tfidf_matrix.shape[1]
    n_components = max(1, min(max_components, n_features))
    lsi = TruncatedSVD(n_components=n_components, random_state=random_state)
    return lsi, lsi.fit_transform(tfidf_matrix)


# _cluster_kmeans: make n_init compatible across sklearn versions
def _cluster_kmeans(lsi_matrix, n_clusters: int, random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
    if lsi_matrix.shape[0] < n_clusters:
        raise ValueError(f"n_clusters ({n_clusters}) > number of documents ({lsi_matrix.shape[0]})")
    try:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km, km.fit_predict(lsi_matrix)


# _project_pca: cap by samples and features
def _project_pca(lsi_matrix, n_components: int = 2, random_state: int = 42):
    n_samples, n_features = lsi_matrix.shape
    n_comp = min(n_components, n_samples, n_features)
    if n_comp < 1:
        # fallback to a trivial single component if absolutely necessary
        n_comp = 1
    pca = PCA(n_components=n_comp, random_state=random_state)
    coords = pca.fit_transform(lsi_matrix)
    # pad to length 2 for UI consistency if needed
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    return pca, coords


def _cosine_to_centroid(lsi_matrix, labels, centroids) -> List[float]:
    return [
        float(cosine_similarity([lsi_matrix[i]], [centroids[labels[i]]])[0][0])
        for i in range(lsi_matrix.shape[0])
    ]

def _top_terms_per_doc(tfidf_matrix, feature_names: List[str], top_k: int = 50):
    arr = tfidf_matrix.toarray()
    details: List[List[Dict[str, Any]]] = []
    for row in arr:
        top_idx = row.argsort()[::-1][:top_k]
        details.append(
            [{"term": feature_names[j], "score": float(row[j])} for j in top_idx if row[j] > 0]
        )
    return details

def _compute_cluster_metrics(tfidf_matrix, labels, doc_count: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for c in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == c]
        sub = tfidf_matrix[idx]
        prevalence = len(idx)
        proportion = (prevalence / doc_count) * 100.0

        tfidf_sum = np.asarray(sub.sum(axis=0)).flatten()
        top_idx = tfidf_sum.argsort()[::-1][:20]
        kv = sub[:, top_idx].T.toarray()

        if kv.shape[0] > 1:
            sims = cosine_similarity(kv)
            vals = [sims[i, j] for i in range(kv.shape[0]) for j in range(i + 1, kv.shape[0])]
            coherence = float(np.mean(vals)) if vals else None
        else:
            coherence = None

        flat = np.asarray(sub.sum(axis=0)).flatten()
        p = flat[flat > 0]
        p = p / p.sum() if p.sum() > 0 else p
        ent = float(scipy_entropy(p, base=2)) if len(p) > 0 else 0.0

        result[c] = {
            "prevalence": prevalence,
            "coherence": coherence,
            "entropy": ent,
            "proportion": proportion,
        }
    return result

def _compute_global_metrics(per_cluster: Dict[str, Any], n_clusters: int, tfidf_details: List[List[Dict[str, Any]]]):
    prevalence = [v["prevalence"] for v in per_cluster.values()]
    proportions = [v["proportion"] for v in per_cluster.values()]
    entropy_overall = float(scipy_entropy(proportions, base=2)) if proportions else 0.0

    unique_terms = set()
    for doc_terms in tfidf_details:
        unique_terms.update([kw["term"] for kw in doc_terms[:50]])
    topic_diversity = len(unique_terms) / float(50 * n_clusters) if n_clusters else 0.0

    coh_vals = [v["coherence"] for v in per_cluster.values() if v["coherence"] is not None]
    average_coherence = float(np.mean(coh_vals)) if coh_vals else None
    average_entropy = float(np.mean([v["entropy"] for v in per_cluster.values()])) if per_cluster else 0.0
    irbo = 1 - (np.std(prevalence) / np.mean(prevalence)) if (prevalence and np.mean(prevalence) > 0) else 0.0

    return {
        "average_coherence": average_coherence,
        "average_entropy": average_entropy,
        "topic_diversity": topic_diversity,
        "irbo": float(irbo),
        "entropy_overall": entropy_overall,
    }


# ---------- Orchestrator ----------
# ---------- Orchestrator ----------
def analyze_corpus_draft_folder(
    drafts_root: Path,
    corpus_id: str,
    n_clusters: int = 15,
) -> Dict[str, Any]:
    """
    Analyze corpus draft (.../c_<id>) and write ONLY what the dashboard needs:
      - analysis_output.json with keys:
          documents, per_cluster, global, corpus, n_documents, n_clusters
    """
    if not corpus_id.startswith("c_"):
        raise ValueError("corpus_id must start with 'c_'")

    corpus_dir = drafts_root / corpus_id

    # Try to load corpus metadata for name + datasets
    meta_name = corpus_id
    meta_datasets: List[str] = []
    meta_path = corpus_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) or {}
            meta_name = meta.get("name", meta_name)
            # datasets were saved when the corpus draft was created
            meta_datasets = "_".join(list(meta.get("datasets") or []))
        except Exception:
            # keep safe fallbacks
            pass

    # Load documents (id, raw_text)
    documents = _load_corpus_documents(corpus_dir)
    texts = [d["raw_text"] for d in documents]
    n_docs = len(documents)

    # pipeline
    vec, tfidf = _build_tfidf(texts)
    lsi, lsi_matrix = _compute_lsi(tfidf)
    km, labels = _cluster_kmeans(lsi_matrix, n_clusters=n_clusters)
    pca, coords = _project_pca(lsi_matrix)
    scores = _cosine_to_centroid(lsi_matrix, labels, km.cluster_centers_)

    feature_names = vec.get_feature_names_out()
    per_doc_terms = _top_terms_per_doc(tfidf, feature_names, top_k=50)

    # documents for dashboard
    doc_outputs: List[Dict[str, Any]] = []
    for i, doc in enumerate(documents):
        start_word = _start_word_for_doc(
            doc_text=doc["raw_text"],
            vec=vec,
            tfidf_row=tfidf[i],           # 1 x V sparse row
            rel_threshold=0.33,           # tune to taste
            min_abs=0.05                  # tune to taste
        )

        doc_outputs.append({
            "id": doc["id"],
            "text": doc["raw_text"].capitalize(),
            "cluster": int(labels[i]),
            "score": float(scores[i]),
            "pca": [float(x) for x in coords[i]],
            "keywords": [kw["term"] for kw in per_doc_terms[i]],
            "start_word": start_word,     
        })


    # metrics
    per_cluster = _compute_cluster_metrics(tfidf, labels, doc_count=n_docs)
    global_metrics = _compute_global_metrics(per_cluster, n_clusters, per_doc_terms)

    # corpus summary: "a mix of the datasets"
    corpus_info = {
        "id": corpus_id,
        "name": meta_name,
        "datasets": meta_datasets,   # e.g., ["d_dataset1", "d_dataset2", ...]
        "source": "merged",
    }

    # combined payload (exactly what the dashboard consumes + requested extras)
    combined_payload = {
        "corpus": corpus_info,
        "n_documents": n_docs,
        "n_clusters": n_clusters,
        "documents": doc_outputs,
        "per_cluster": {str(k): v for k, v in per_cluster.items()},
        "global": global_metrics,
    }

    # persist only this file
    out_path = corpus_dir / "analysis_output.json"
    out_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")

    return combined_payload



def _start_word_for_doc(
    doc_text: str,
    vec: TfidfVectorizer,
    tfidf_row,  # 1 x V sparse row for this doc
    rel_threshold: float = 0.33,   # e.g., at least 33% of the doc’s max TF–IDF
    min_abs: float = 0.05          # and at least this absolute TF–IDF
) -> str | None:
    """
    Return the first token (by original order, after TF–IDF analyzer processing)
    whose TF–IDF score meets a 'relatively higher' threshold.

    Threshold rule: score >= max(rel_threshold * max_score_in_doc, min_abs)
    """
    # Use the SAME preprocessing/tokenization as vec
    analyzer = vec.build_analyzer()
    tokens = analyzer(doc_text)

    # Build quick accessors
    vocab = vec.vocabulary_                 # term -> index
    row_dense = tfidf_row.toarray().ravel() # TF–IDF scores for this doc
    if row_dense.size == 0:
        return None

    doc_max = float(row_dense.max()) if row_dense.size else 0.0
    if doc_max <= 0.0:
        return None

    cutoff = max(rel_threshold * doc_max, min_abs)

    for tok in tokens:
        j = vocab.get(tok)
        if j is None:
            continue
        score = float(row_dense[j])
        if score >= cutoff:
            return tok

    # Fallback: first non-zero token if nothing met the threshold
    for tok in tokens:
        j = vocab.get(tok)
        if j is not None and row_dense[j] > 0.0:
            return tok

    return None

def _save_training_payload(corpus_dir: Path, payload: Dict[str, Any]) -> Path:
    """
    Save the training payload (parameters passed into the analysis endpoint)
    into the draft folder as training_payload.json
    """
    out_path = corpus_dir / "training_payload.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

