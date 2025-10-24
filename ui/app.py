from flask import Flask, request, jsonify, render_template, current_app, send_from_directory
import os
from util import *
import requests
from flask import session
import json
from flask import render_template, Response
from pathlib import Path

server = Flask(__name__)

server.secret_key = 'your-secret-key'

API = os.getenv("API_BASE_URL", "http://api:8000")

# @daniel-stephens: maybe we can remove this and move to a config or something
modelurl = f"{API}/model"
inferurl = f"{API}/infer/json"
topicinfourl = f"{API}/queries/model-info"
textinfourl = f"{API}/queries/thetas-by-docs-ids"


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


@server.route("/data/create/corpus/", methods=["POST"])
def create_corpus():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "No JSON payload received"}), 400

    datasets = payload.get("datasets", [])  # List of dictionaries with dataset IDs
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
                    mimetype=upstream.headers.get("Content-Type", "application/json"),
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

    current_app.logger.info("Payload sent to /data/corpora: %s", corpus_payload)

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
                mimetype=upstream.headers.get("Content-Type", "application/json"),
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
    payload = request.get_json(silent=True) or {}
    payload.setdefault("n_clusters", 15)

    try:
        upstream = requests.post(
            f"{API}/train/corpus/{corpus_id}/tfidf/",
            json=payload,
            timeout=(3.05, 60),
        )
        upstream.raise_for_status()
    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
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


@server.route("/training/", methods=["GET"])
def training_page():
    corpus_id = request.args.get("corpus_id")  # e.g. "c_93e37bf9"
    return render_template("training.html", corpus_id=corpus_id)


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


@server.route("/train/trainingInfo/<corpus_id>/", methods=["GET"])
def replay_training_from_saved(corpus_id):
    try:
        # 1) Fetch saved training params/payload from upstream
        up = requests.get(
            f"{API}/train/trainingInfo/{corpus_id}/", timeout=(3.05, 30))
        up.raise_for_status()

        try:
            payload = up.json()
        except ValueError:
            return jsonify({"error": "Upstream returned non-JSON training payload"}), 502

        # --- FIX #1: unwrap training_config if upstream wraps it ---
        if isinstance(payload, dict) and "training_config" in payload and isinstance(payload["training_config"], dict):
            payload = payload["training_config"]

        # --- FIX #2 (defensive): normalize data records to TrainRequest shape ---
        # Expected: payload["data"] is a list of {id: ..., raw_text: ...}
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            normalized = []
            for rec in payload["data"]:
                if not isinstance(rec, dict):
                    continue
                rid = rec.get("id")
                # some sources use "text" instead of "raw_text"
                rtxt = rec.get("raw_text", rec.get("text"))
                if rid is not None and rtxt is not None:
                    normalized.append({"id": str(rid), "raw_text": str(rtxt)})
            payload["data"] = normalized

        # 2) Forward (POST) to upstream training endpoint
        headers = {}
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

        tr = requests.post(
            f"{API}/train/json",
            json=payload,
            headers=headers,
            timeout=(3.05, 30),
        )

        # If validation failed, surface the upstream error body to help debugging
        if tr.status_code == 422:
            # Return exactly what upstream said (Pydantic error details)
            return Response(
                tr.content,
                status=tr.status_code,
                mimetype=tr.headers.get("Content-Type", "application/json"),
            )

        # 3) Build Flask response mirroring upstream body/status
        resp = Response(
            tr.content,
            status=tr.status_code,
            mimetype=tr.headers.get("Content-Type", "application/json"),
        )
        # Propagate Location header (FastAPI sets: /status/jobs/<job_id>)
        loc = tr.headers.get("Location")
        if loc:
            resp.headers["Location"] = loc

        return resp

    except requests.Timeout:
        return jsonify({"error": "Upstream timeout"}), 504
    except requests.HTTPError as e:
        # Bubble up details from the initial GET
        return jsonify({
            "error": f"Upstream GET returned {e.response.status_code}",
            "detail": e.response.text
        }), e.response.status_code
    except requests.RequestException as e:
        return jsonify({"error": f"Upstream connection error: {e}"}), 502


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


# This function gets the configuration file for the Models
@server.route("/model-config")
def get_model_config():
    with open("static/config/model_config.json") as f:
        return jsonify(json.load(f))


# This function gets the model registry
@server.route("/model-registry")
def get_model_registry():
    with open("static/config/modelRegistry.json") as f:
        return jsonify(json.load(f))


##########################################################
# MODEL-RELATED ROUTES
##########################################################
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
                models_response = requests.get(f"{API}/data/corpora/{corpus_id}/models", timeout=10)
                models_response.raise_for_status()
                corpus["models"] = models_response.json()
            except requests.Timeout:
                current_app.logger.error("Timeout fetching models for corpus ID %s", corpus_id)
                corpus["models"] = {"error": "Timeout fetching models"}
            except requests.RequestException as e:
                current_app.logger.error("Error fetching models for corpus ID %s: %s", corpus_id, str(e))
                corpus["models"] = {"error": "Failed to fetch models", "detail": str(e)}

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


@server.route("/dashboard/<modelId>", methods=["GET", "POST"])
def dashboard(modelId):

    modelId = modelId
    return render_template("dashboard.html", model_id=modelId)


@server.route("/get-dashboard-data", methods=["GET", "POST"])
def get_data():
    data_path = Path("dashboardData.json")
    # Read the JSON file from disk and return it
    with data_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    # jsonify sets the correct Content-Type and handles UTF-8 safely
    return jsonify(payload)


@server.post("/save-settings")
def save_settings():
    payload = request.get_json(silent=True) or {}
    server.logger.info("Dummy /save-settings received: %s", payload)
    return jsonify({"ok": True, "echo": payload}), 200
