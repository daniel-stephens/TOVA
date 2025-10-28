import json
import logging
import os
from pathlib import Path
from time import time

import requests
from dashboard_utils import pydantic_to_dict, to_dashboard_bundle
from flask import (Flask, Response, current_app, jsonify, render_template,
                   request, session)
from util import *

server = Flask(__name__)
server.logger.setLevel(logging.INFO)

server.secret_key = 'your-secret-key'

DRAFTS_SAVE = Path(os.getenv("DRAFTS_SAVE", "/data/drafts"))  # this needs to be removed
API = os.getenv("API_BASE_URL", "http://api:8000")

_CACHE_MODELS = {}
_CACHE_CORPORA = {}
_CACHE_MODEL_INFO = {}      # key: model_path
_CACHE_THETAS = {}          # key: (model_path, doc_id)
_CACHE_THETAS_BULK = {}     # key: (model_path, tuple(sorted_doc_ids))
CACHE_TTL = 600


def _cache_get(cache, key):
    entry = cache.get(key)
    if not entry:
        return None, "miss"
    if time() - entry["ts"] > CACHE_TTL:
        return None, "expired"
    return entry["data"], "hit"


def _cache_set(cache, key, data):
    cache[key] = {"ts": time(), "data": data}

# 1. Home Page
@server.route('/')
def home():
    return render_template('homepage.html')


@server.route('/check-backend')
def check_backend():
    try:
        response = requests.get(f"{API}/health")
        if response.status_code == 200:
            return jsonify(status="success", message="Backend is healthy"), 200
        else:
            return jsonify(status="error", message="Backend is not healthy"), 500
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500


@server.route("/load-data-page/")
def load_data_page():
    return render_template('index.html')


@server.route("/load-corpus-page/")
def load_corpus_page():
    return render_template('manageCorpora.html')


@server.route("/data/create/corpus/", methods=["POST"])
def create_corpus():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "No JSON payload received"}), 400

    # List of dictionaries with dataset IDs
    datasets = payload.get("datasets", [])
    # @TODO: daniel-stephens: this is fixed in the html code and it should not always be a list of one dictionary
    datasets_lst = []

    for el in datasets:
        try:
            upstream = requests.get(
                f"{API}/data/datasets/{el['id']}",
                timeout=(3.05, 30),
            )
            if not upstream.ok:
                return Response(
                    upstream.content,
                    status=upstream.status_code,
                    mimetype=upstream.headers.get(
                        "Content-Type", "application/json"),
                )
            datasets_lst.append(upstream.json())
        except requests.Timeout:
            return jsonify({"error": "Upstream timeout"}), 504
        except requests.RequestException as e:
            return jsonify({"error": f"Upstream connection error: {e}"}), 502

    corpus_payload = {
        "name": payload.get("corpus_name", ""),
        "description": f"Corpus from datasets {', '.join([d['id'] for d in datasets])}",
        "datasets": datasets_lst,
    }

    current_app.logger.info(
        "Payload sent to /data/corpora: %s", corpus_payload)

    try:
        upstream = requests.post(
            f"{API}/data/corpora",
            json=corpus_payload,
            timeout=(3.05, 30),
        )
        if not upstream.ok:
            return Response(
                upstream.content,
                status=upstream.status_code,
                mimetype=upstream.headers.get(
                    "Content-Type", "application/json"),
            )

        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


@server.route("/data/create/dataset/", methods=["POST"])
def create_dataset():

    payload = request.get_json(silent=True) or {}
    metadata = payload.get("metadata", {})
    data = payload.get("data", {})
    documents = data.get("documents", [])
    owner_id = payload.get("owner_id")

    dataset_payload = {
        "name": metadata.get("name", ""),
        "description": metadata.get("description", ""),
        "owner_id": owner_id,
        "documents": documents,
        "metadata": metadata
    }

    try:
        upstream = requests.post(
            f"{API}/data/datasets",
            json=dataset_payload,
            timeout=(3.05, 30),
        )
        if not upstream.ok:
            return Response(
                upstream.content,
                status=upstream.status_code,
                mimetype=upstream.headers.get(
                    "Content-Type", "application/json"),
            )
        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )

    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


@server.route("/train/corpus/<corpus_id>/tfidf/", methods=["POST"])
def train_tfidf_corpus(corpus_id):

    # get config from browser
    payload = request.get_json(silent=True) or {}
    try:
        n_clusters = int(payload.get("n_clusters") or 15)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid n_clusters"}), 400

    # fetch corpus from backend
    try:
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}", timeout=(3.05, 60))
        up.raise_for_status()
        corpus = up.json()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout fetching corpus"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream error fetching corpus: {e}"}), 502

    documents = [
        {"id": str(d.get("id")), "raw_text": str(d.get("text", ""))}
        for d in (corpus.get("documents") or [])
    ]
    if not documents:
        return jsonify({"error": "No documents found in corpus"}), 400

    # call TF-IDF upstream
    upstream_payload = {"n_clusters": n_clusters, "documents": documents}
    try:
        upstream = requests.post(
            f"{API}/train/corpus/{corpus_id}/tfidf/",
            json=upstream_payload,
            timeout=(3.05, 120),
        )
        upstream.raise_for_status()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout calling TF-IDF"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502

    try:
        return jsonify(upstream.json()), upstream.status_code
    except ValueError:
        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )


@server.get("/training/")
def training_page_get():
    return render_template("training.html")


@server.route("/get-training-session", methods=["GET"])
def get_training_session():
    corpus_id = session.get("corpus_id")
    tmReq = session.get("tmReq")

    if not corpus_id or not tmReq:
        return jsonify({"error": "Training session data not found"}), 404

    return jsonify({"corpus_id": corpus_id, "tmReq": tmReq}), 200


@server.route("/train/corpus/<corpus_id>/tfidf/", methods=["GET"])
def proxy_corpus_tfidf(corpus_id):
    try:
        up = requests.get(
            f"{API}/train/corpus/{corpus_id}/tfidf/", timeout=(3.05, 30))
        return Response(up.content, status=up.status_code,
                        mimetype=up.headers.get("Content-Type", "application/json"))
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


@server.route("/corpus/<corpus_id>/tfidf/", methods=["GET"])
def get_tfidf_data(corpus_id):
    try:
        # forward any query params (e.g., n_clusters=…)
        upstream = requests.get(
            f"{API}/train/corpus/{corpus_id}/tfidf/",
            params=request.args,          # forward ?n_clusters=…
            timeout=(3.05, 60),
        )
        upstream.raise_for_status()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502

    # Prefer JSON if the upstream sent JSON
    try:
        return jsonify(upstream.json()), upstream.status_code
    except ValueError:
        # Not JSON? Return raw body with the upstream content-type.
        return Response(
            upstream.content,
            status=upstream.status_code,
            mimetype=upstream.headers.get("Content-Type", "application/json"),
        )


@server.post("/training/start")
def training_start():
    payload = request.get_json(silent=True) or {}
    corpus_id = payload.get("corpus_id")
    model = payload.get("model")
    model_name = payload.get("model_name")
    training_params = payload.get("training_params") or {}

    if not corpus_id or not model or not model_name:
        return jsonify({"error": "Missing corpus_id/model/model_name"}), 400

    # get corpus
    try:
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}", timeout=(3.05, 60))
        up.raise_for_status()
        corpus = up.json()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout fetching corpus"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream error fetching corpus: {e}"}), 502

    docs = [
        {"id": str(d.get("id")), "text": str(d.get("text", ""))}
        for d in (corpus.get("documents") or [])
    ]
    if not docs:
        return jsonify({"error": "No documents found in corpus"}), 400

    # tr request
    tm_req = {
        "model": model,
        "corpus_id": corpus_id,
        "data": docs,
        "id_col": "id",
        "text_col": "text",
        "do_preprocess": False,
        "training_params": training_params,
        "config_path": "static/config/config.yaml",
        "model_name": model_name,
    }

    # training upstream
    try:
        tr = requests.post(f"{API}/train/json",
                           json=tm_req, timeout=(3.05, 120))
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout starting training"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream error starting training: {e}"}), 502

    if tr.status_code >= 400:
        return Response(
            tr.content,
            status=tr.status_code,
            mimetype=tr.headers.get("Content-Type", "application/json"),
        )

    # Extract job_id (and optional Location)
    job_id = None
    try:
        body = tr.json()
        job_id = (body or {}).get("job_id")
    except Exception:
        body = {}

    loc = tr.headers.get("Location")
    return jsonify({
        "job_id": job_id,
        "status_url": loc or (f"/status/jobs/{job_id}" if job_id else None),
        "corpus_id": corpus_id,
    }), 200

# Accept either /status/jobs/<job_id> or /status/jobs/?job_id=...


@server.route("/status/jobs/<job_id>", methods=["GET"])
@server.route("/status/", methods=["GET"])
def get_status(job_id=None):
    # 1) Resolve job_id from path or query/header
    job_id = job_id or request.args.get(
        "job_id") or request.headers.get("X-Job-Id")
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400

    # 2) Optional auth passthrough
    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    # 3) Call upstream status endpoint
    upstream_url = f"{API}/status/jobs/{job_id}"
    try:
        up = requests.get(upstream_url, headers=headers, timeout=(3.05, 30))
        # Proxy upstream response body + status + content-type
        return Response(
            up.content,
            status=up.status_code,
            mimetype=up.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


# # 2. Select Model Page
@server.route('/model')
def loadModel():

    return render_template('loadModel.html')


@server.route("/getUniqueCorpusNames", methods=["GET"])
def get_unique_corpus_names():
    """
    Return a deduped (case-insensitive), A→Z list of corpus names.
    Prefers the most recent 'created_at' when duplicates exist.
    """
    try:
        r = requests.get(f"{API}/data/corpora", timeout=10)
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        # note: server.logger
        server.logger.exception("Failed to fetch drafts from upstream")
        return jsonify({"error": "Upstream drafts service failed"}), 502

    pick = {}  # lower(name) -> {"name": original, "created_at": ts}
    for d in items or []:
        meta = (d or {}).get("metadata") or {}
        name = (meta.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        created = meta.get("created_at") or ""
        if key not in pick or created > pick[key]["created_at"]:
            pick[key] = {"name": name, "created_at": created}

    names = sorted((v["name"] for v in pick.values()), key=str.casefold)
    return jsonify(names), 200


@server.route("/getAllCorpora", methods=["GET"])
def getAllCorpora():
    """
    Pulls all corpora (draft or permanent storage) from the upstream API.
    """

    try:
        r = requests.get(f"{API}/data/corpora", timeout=10)
        r.raise_for_status()
        items = r.json()
    except requests.RequestException:
        # note: server.logger
        server.logger.exception("Failed to fetch drafts from upstream")
        return jsonify({"error": "Upstream drafts service failed"}), 502

    def _norm_corpus(c):
        location = c.get("location")
        if location == "database":
            is_draft = False
        else:
            is_draft = True
        return {
            "id": c.get("id"),
            "name": c.get("metadata", {}).get("name", ""),
            "is_draft": is_draft,
            "created_at": c.get("metadata", {}).get("created_at", ""),
        }

    corpora = [_norm_corpus(c) for c in items]
    corpora.sort(key=lambda x: (x["name"].lower(), not x["is_draft"]))

    return jsonify(corpora), 200


@server.route("/getCorpus/<corpus_id>")
def get_corpus(corpus_id):
    try:
        response = requests.get(f"{API}/data/corpora/{corpus_id}", timeout=10)
        response.raise_for_status()
        return Response(
            response.content,
            status=response.status_code,
            mimetype=response.headers.get("Content-Type", "application/json"),
        )
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.RequestException as e:
        current_app.logger.error(f"Failed to fetch corpus {corpus_id}: {e}")
        return jsonify({"error": f"Failed to fetch corpus {corpus_id}: {e}"}), 502

##########################################################
# MODEL-RELATED ROUTES
##########################################################

@server.route("/model-registry")
def get_model_registry():
    with open("static/config/modelRegistry.json") as f:
        return jsonify(json.load(f))


@server.route("/trained-models")
def trained_models():
    return render_template("trained_models.html")


def fetch_trained_models():
    try:
        # Fetch corpora
        corpora_response = requests.get(f"{API}/data/corpora", timeout=10)
        corpora_response.raise_for_status()
        corpora = corpora_response.json()
        current_app.logger.info("Corpora response: %s", corpora)

        # Fetch models for each corpus
        for corpus in corpora:
            corpus_id = corpus.get("id")
            if not corpus_id:
                current_app.logger.warning("Corpus without ID: %s", corpus)
                continue

            try:
                models_response = requests.get(
                    f"{API}/data/corpora/{corpus_id}/models", timeout=10)
                models_response.raise_for_status()
                corpus["models"] = models_response.json()
            except requests.Timeout:
                current_app.logger.error(
                    "Timeout fetching models for corpus ID %s", corpus_id)
                corpus["models"] = {"error": "Timeout fetching models"}
            except requests.RequestException as e:
                current_app.logger.error(
                    "Error fetching models for corpus ID %s: %s", corpus_id, str(e))
                corpus["models"] = {
                    "error": "Failed to fetch models", "detail": str(e)}

        return corpora

    except requests.Timeout:
        current_app.logger.error("Timeout fetching corpora")
        raise
    except requests.RequestException as e:
        current_app.logger.error("Error fetching corpora: %s", str(e))
        raise
    except ValueError:
        current_app.logger.error("Invalid JSON response from corpora endpoint")
        raise


@server.route("/get-trained-models", methods=["GET"])
def get_trained_models():
    try:
        corpora = fetch_trained_models()
        return jsonify(corpora), 200
    except requests.Timeout:
        return jsonify({"error": "Upstream request timed out"}), 504
    except requests.RequestException as e:
        return jsonify({"error": "Upstream request failed", "detail": str(e)}), 502
    except ValueError:
        return jsonify({"error": "Invalid JSON from upstream"}), 502


@server.route("/get-models-names", methods=["GET"])
def get_model_names():
    try:
        corpora = fetch_trained_models()

        names = []
        for corpus in corpora:
            models = corpus.get("models", [])
            for model in models:
                meta = model.get("metadata") or {}
                tr = meta.get("tr_params") or {}
                name = tr.get("model_name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())

        # Deduplicate
        seen = set()
        unique = []
        for n in names:
            key = n.lower()
            if key not in seen:
                seen.add(key)
                unique.append(n)

        return jsonify({"models": unique})

    except requests.Timeout:
        return jsonify({"error": "Upstream request timed out"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream request failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@server.route("/delete-model", methods=["POST"])
def delete_model():
    """
    Body (JSON):
      { "model_id": "...", "corpus_id": "..." }  # corpus_id optional
    If corpus_id is missing, we search all corpora to find the model, then delete.
    """

##########################################################
# DASHBOARD-RELATED ROUTES
##########################################################


@server.route("/dashboard", methods=["POST"])
def dashboard():
    model_id = request.form.get("model_id", "")
    # TODO: temporary; the backend should know whether is BBDD or draft
    model_path = DRAFTS_SAVE / model_id

    return render_template(
        "dashboard.html",
        model_id=model_id,
        model_name=model_path
    )


@server.route("/get-dashboard-data", methods=["POST"])
def proxy_dashboard_data():
    t0 = time()
    payload = request.get_json(silent=True) or {}
    payload.setdefault("config_path", "static/config/config.yaml")

    model_id = payload.get("model_id", "")
    model_path = payload.get("model_path", "")
    server.logger.info(
        "GET-DASHBOARD start model_id=%s model_path=%s", model_id, model_path)

    if not model_id:
        server.logger.warning("GET-DASHBOARD missing model_id")
        return jsonify({"error": "model_id is required"}), 400

    # model metadata
    try:
        up0 = time()
        up = requests.get(f"{API}/data/models/{model_id}", timeout=(3.05, 60))
        up.raise_for_status()
        model_metadata = up.json()
        server.logger.info("GET-DASHBOARD model meta ok status=%s dt=%.3fs",
                           up.status_code, time() - up0)
    except requests.Timeout:
        server.logger.exception("GET-DASHBOARD timeout fetching model")
        return jsonify({"error": "Upstream timeout fetching model"}), 504
    except requests.RequestException as e:
        server.logger.exception("GET-DASHBOARD error fetching model: %s", e)
        return jsonify({"error": f"Upstream error fetching model: {e}"}), 502

    corpus_id = model_metadata.get("corpus_id", "")
    if not corpus_id:
        server.logger.error("GET-DASHBOARD model missing corpus_id")
        return jsonify({"error": "Model missing corpus_id"}), 400

    # corpus data
    try:
        up0 = time()
        up = requests.get(
            f"{API}/data/corpora/{corpus_id}", timeout=(3.05, 60))
        up.raise_for_status()
        corpus_training_data = up.json()
        server.logger.info("GET-DASHBOARD corpus ok status=%s dt=%.3fs docs=%s",
                           up.status_code, time() - up0,
                           len(corpus_training_data.get("documents", []) or []))
    except requests.Timeout:
        server.logger.exception("GET-DASHBOARD timeout fetching corpus")
        return jsonify({"error": "Upstream timeout fetching corpus"}), 504
    except requests.RequestException as e:
        server.logger.exception("GET-DASHBOARD error fetching corpus: %s", e)
        return jsonify({"error": f"Upstream error fetching corpus: {e}"}), 502

    # model info
    model_info_key = model_path or f"__by_id__:{model_id}"
    cached_info, info_state = _cache_get(_CACHE_MODEL_INFO, model_info_key)
    if cached_info:
        raw_model_info = cached_info
        server.logger.debug("GET-DASHBOARD model-info cache %s", info_state)
    else:
        try:
            up0 = time()
            r = requests.post(f"{API}/queries/model-info",
                              json=payload, timeout=60)
            r.raise_for_status()
            raw_model_info = r.json()
            _cache_set(_CACHE_MODEL_INFO, model_info_key, raw_model_info)
            server.logger.info(
                "GET-DASHBOARD model-info fetched status=%s dt=%.3fs (cached)", r.status_code, time() - up0)
        except requests.HTTPError:
            try:
                return jsonify(r.json()), r.status_code
            except Exception:
                return jsonify({"detail": r.text}), r.status_code
        except Exception as e:
            server.logger.exception(
                "GET-DASHBOARD model-info proxy error: %s", e)
            return jsonify({"detail": f"Proxy error: {e}"}), 502

    # Doc list and thetas
    docs_list_raw = corpus_training_data.get("documents", []) or []
    docs_list = [d if isinstance(d, dict) else pydantic_to_dict(d)
                 for d in docs_list_raw]
    doc_ids = [str(d.get("id")) for d in docs_list if d.get("id") is not None]

    sorted_key = tuple(sorted(doc_ids))
    bulk_key = (model_path, sorted_key)
    cached_bulk, bulk_state = _cache_get(_CACHE_THETAS_BULK, bulk_key)
    if cached_bulk:
        doc_thetas = cached_bulk
        server.logger.debug(
            "GET-DASHBOARD thetas bulk cache %s | docs=%d", bulk_state, len(doc_ids))
    else:
        payload_thetas = {
            "docs_ids": ",".join(doc_ids),
            "model_path": model_path,
        }
        try:
            up0 = time()
            r = requests.post(
                f"{API}/queries/thetas-by-docs-ids", json=payload_thetas, timeout=60)
            r.raise_for_status()
            doc_thetas = r.json()
            _cache_set(_CACHE_THETAS_BULK, bulk_key, doc_thetas)
            server.logger.info("GET-DASHBOARD thetas fetched status=%s dt=%.3fs (cached) docs=%d",
                               r.status_code, time() - up0, len(doc_ids))
        except requests.HTTPError:
            try:
                return jsonify(r.json()), r.status_code
            except Exception:
                return jsonify({"detail": r.text}), r.status_code
        except Exception as e:
            server.logger.exception("GET-DASHBOARD thetas proxy error: %s", e)
            return jsonify({"detail": f"Proxy error: {e}"}), 502

    # Build dashboard bundle
    try:
        model_training_corpus = pydantic_to_dict(corpus_training_data) or {}
        bundle = to_dashboard_bundle(
            raw_model_info,
            Path(payload.get("model_path", "")),
            model_metadata=model_metadata.get("metadata", {}),
            model_training_corpus=model_training_corpus,
            doc_thetas=doc_thetas,
            logger=server.logger,
        )
        server.logger.info("GET-DASHBOARD done dt=%.3fs", time() - t0)
        return jsonify(bundle), 200
    except Exception as e:
        server.logger.exception("GET-DASHBOARD bundling error: %s", e)
        return jsonify({"error": f"Failed to build dashboard data: {e}"}), 500


@server.route("/text-info", methods=["POST"])
def text_info():
    """
    Get text and model information for a given document according to a topic model to display when a document is clicked through its row click in dashboard.html.

    Returns:
      {
        "theme": "<label to show>",
        "top_themes": [
          {"theme_id": <int>, "label": "", "score": <float>, "keywords": ""}
        ],
        "rationale": "",
        "text": "<the actual document text>"
      }
    """
    t0 = time()
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id", "")
    model_path = payload.get("model_path", "")
    doc_id = str(payload.get("document_id", "")).strip()

    server.logger.info("TEXT-INFO start model_id=%s model_path=%s doc_id=%s",
                       model_id, model_path, doc_id)

    if not model_id or not doc_id:
        server.logger.warning("TEXT-INFO missing required fields")
        return jsonify({"detail": "model_id and document_id are required."}), 400

    # model meta info (cached by _CACHE_MODELS)
    model_entry = _CACHE_MODELS.get(model_id)
    if not model_entry or (time() - model_entry["ts"] > CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{API}/data/models/{model_id}", timeout=(3.05, 30))
            up.raise_for_status()
            model = up.json()
            _CACHE_MODELS[model_id] = {"ts": time(), "data": model}
            server.logger.info("TEXT-INFO model meta fetched status=%s dt=%.3fs (cached)",
                               up.status_code, time() - up0)
        except requests.RequestException as e:
            server.logger.exception("TEXT-INFO fetch model error: %s", e)
            return jsonify({"detail": f"Failed to fetch model info: {e}"}), 502
    else:
        model = model_entry["data"]
        server.logger.debug("TEXT-INFO model meta cache hit")

    corpus_id = model.get("corpus_id", "")
    if not corpus_id:
        server.logger.error("TEXT-INFO model missing corpus_id")
        return jsonify({"detail": "Model missing corpus_id."}), 400

    # corpus info (cached by _CACHE_CORPORA)
    corpus_entry = _CACHE_CORPORA.get(corpus_id)
    if not corpus_entry or (time() - corpus_entry["ts"] > CACHE_TTL):
        try:
            up0 = time()
            up = requests.get(
                f"{API}/data/corpora/{corpus_id}", timeout=(3.05, 60))
            up.raise_for_status()
            corpus = up.json()
            _CACHE_CORPORA[corpus_id] = {"ts": time(), "data": corpus}
            server.logger.info("TEXT-INFO corpus fetched status=%s dt=%.3fs (cached)",
                               up.status_code, time() - up0)
        except requests.RequestException as e:
            server.logger.exception("TEXT-INFO fetch corpus error: %s", e)
            return jsonify({"detail": f"Failed to fetch corpus info: {e}"}), 502
    else:
        corpus = corpus_entry["data"]
        server.logger.debug("TEXT-INFO corpus cache hit")

    docs = corpus.get("documents", [])
    doc_text = next((d.get("text", "")
                    for d in docs if str(d.get("id")) == doc_id), "")
    if not doc_text:
        doc_text = f"(Text not found for document {doc_id})"

    # thetas for this document (cached by _CACHE_THETAS)
    theta_key = (model_path, doc_id)
    cached_theta, theta_state = _cache_get(_CACHE_THETAS, theta_key)
    if cached_theta:
        thetas_by_doc = {doc_id: cached_theta}
        server.logger.debug(
            "TEXT-INFO thetas cache %s for doc_id=%s", theta_state, doc_id)
    else:
        try:
            up0 = time()
            r = requests.post(
                f"{API}/queries/thetas-by-docs-ids",
                json={
                    "model_path": model_path,
                    "config_path": "static/config/config.yaml",
                    "docs_ids": doc_id,
                },
                timeout=60
            )
            r.raise_for_status()
            thetas_by_doc = r.json()
            _cache_set(_CACHE_THETAS, theta_key,
                       thetas_by_doc.get(doc_id) or {})
            server.logger.info("TEXT-INFO thetas fetched status=%s dt=%.3fs (cached)",
                               r.status_code, time() - up0)
        except requests.RequestException as e:
            server.logger.exception("TEXT-INFO fetch thetas error: %s", e)
            return jsonify({"detail": f"Failed to fetch thetas: {e}"}), 502

    doc_thetas = thetas_by_doc.get(doc_id) or {}

    # model-info (cached by _CACHE_MODEL_INFO)
    model_info_key = model_path or f"__by_id__:{model_id}"
    cached_info, info_state = _cache_get(_CACHE_MODEL_INFO, model_info_key)
    if cached_info:
        raw_model_info = cached_info
        server.logger.debug("TEXT-INFO model-info cache %s", info_state)
    else:
        try:
            up0 = time()
            r = requests.post(f"{API}/queries/model-info",
                              json=payload, timeout=60)
            r.raise_for_status()
            raw_model_info = r.json()
            _cache_set(_CACHE_MODEL_INFO, model_info_key, raw_model_info)
            server.logger.info(
                "TEXT-INFO model-info fetched status=%s dt=%.3fs (cached)", r.status_code, time() - up0)
        except requests.HTTPError:
            try:
                return jsonify(r.json()), r.status_code
            except Exception:
                return jsonify({"detail": r.text}), r.status_code
        except Exception as e:
            server.logger.exception("TEXT-INFO model-info proxy error: %s", e)
            return jsonify({"detail": f"Proxy error: {e}"}), 502

    if not doc_thetas:
        server.logger.info(
            "TEXT-INFO no thetas for doc_id=%s dt=%.3fs", doc_id, time() - t0)
        return jsonify({
            "theme": "Unknown",
            "top_themes": [],
            "rationale": "",
            "text": doc_text
        })

    # top themes
    topics_info = raw_model_info.get("Topics Info", {}) or {}
    top_themes = sorted(
        (
            {
                "theme_id": int(k),
                "label": f"Topic {k}",
                "score": float(v),
                "keywords": (topics_info.get(str(k), {}) or {}).get("Keywords", "")
            }
            for k, v in doc_thetas.items() if v is not None
        ),
        key=lambda x: x["score"],
        reverse=True,
    )
    top_theme = top_themes[0] if top_themes else {
        "theme_id": None, "label": "Unknown", "score": 0, "keywords": ""}

    server.logger.info("TEXT-INFO done doc_id=%s top_theme=%s dt=%.3fs",
                       doc_id, top_theme.get("theme_id"), time() - t0)

    return jsonify({
        "theme": top_theme["label"],
        "top_themes": top_themes,
        "rationale": "",
        "text": doc_text
    })


@server.post("/save-settings")
def save_settings():
    payload = request.get_json(silent=True) or {}
    server.logger.info("Dummy /save-settings received: %s", payload)
    return jsonify({"ok": True, "echo": payload}), 200
