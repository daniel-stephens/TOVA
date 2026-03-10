# TOVA — Detailed Feature Reference

This document describes the features included in **TOVA Version 1**. For version scope and exclusions (LLM-based topic models, topic-model chatbot), see [VERSION.md](VERSION.md).

---

## 1. Overview

**TOVA** (Topic Visualization & Analysis) provides a unified interface to train, load, infer, and visualize topic models. Core logic lives in shared Python code and is exposed via:

- **CLI** — Typer-based commands for training, inference, and model queries.
- **REST API** — FastAPI backend used by the web UI and external clients.
- **Web UI** — Flask-based dashboard for corpora, training, model exploration, and inference.

All topic models follow a common base interface (`BaseTMModel` / `TradTMmodel`) and are registered in `static/config/modelRegistry.json`, so new model types can be added without changing the dispatcher.

---

## 2. Topic modeling (Version 1)

### 2.1 Supported models

Version 1 supports **traditional (non-LLM) topic models** only:

| Model           | Registry key   | Description                          |
|----------------|----------------|--------------------------------------|
| Tomotopy LDA    | `tomotopyLDA`  | LDA implementation (tomotopy).      |
| CTM            | `CTM`          | Contextual Topic Model.              |

Training parameters (e.g. `num_topics`, `num_iters`, `alpha`) are configurable via `static/config/config.yaml` and can be overridden per request in the API or UI.

### 2.2 Optional LLM support for traditional models

For traditional models only, TOVA can use an LLM to:

- **Label topics** — Generate a short label from topic keywords and representative documents (`do_labeller`, `labeller_prompt`).
- **Summarize topics** — Generate a short summary per topic (`do_summarizer`, `summarizer_prompt`).

LLM connectivity is configured in `config.yaml` under `llm` (e.g. OpenAI/Azure via API key, or Ollama via host). This is **not** the same as “LLM-based topic models” (TopicGPT, OpenTopicRAG), which are out of scope for Version 1.

---

## 3. Preprocessing

- **Pipeline**: Implemented in `tova.preprocessing.tm_preprocessor.TMPreprocessor`, configured by the `tm_preprocessor` section of `config.yaml`.
- **Spacy**: Lemmatization, POS filtering (`valid_pos`), and optional stopwords. Model configurable (e.g. `en_core_web_sm`).
- **Vectorization**: Count/TF-IDF options (e.g. `max_features`, `min_df`, `max_df`, `ngram_range`).
- **Embeddings**: Optional sentence embeddings (e.g. SentenceTransformers `all-MiniLM-L6-v2`) for models that need them (e.g. CTM).
- Preprocessing can be enabled/disabled when training; if disabled, input is expected to already contain a suitable text/lemma column.

---

## 4. Training

### 4.1 Entry points

- **CLI**: `python -m tova.cli.main train run --model <name> --data <path> --text-col <col> --output <path> …`
- **API**: `POST /train/file` or `POST /train/corpus/<corpus_id>` (and related endpoints) with JSON body (model name, data reference, training params, output path).
- **Web UI**: Create or select a corpus, choose model and parameters, start training. Job status is polled via `GET /status/jobs/<job_id>`.

### 4.2 Behavior

- Training runs in a background job; the API returns a `job_id` and optional `Location` header.
- Progress and completion are reported via job status (and optionally SSE logs).
- Output is written to a **draft** directory (e.g. under `data/drafts` or `DRAFTS_SAVE`). The user can then “promote” the draft to the persistent store (indexing to Solr/DB), which is a separate step.

### 4.3 TF-IDF / corpus exploration

- **TF-IDF for a corpus**: Endpoints to compute and retrieve TF-IDF (and related) information for a corpus, used by the UI for exploration and training setup (e.g. `GET/POST /train/corpus/<corpus_id>/tfidf/`).

---

## 5. Inference

- **Purpose**: Assign topic weights (thetas) to new documents using a trained model.
- **CLI**: `python -m tova.cli.main infer run --data <path> --text-col <col> --model-path <path> …`
- **API**: `POST /infer/file` or `POST /infer/json` with model reference and document list.
- **Web UI**: “Infer on text” (or similar) flow: user provides text (or selects corpus/dataset), backend runs inference and returns thetas (and optionally other diagnostics).
- Inference uses the same preprocessing configuration as training when applicable.

---

## 6. Model queries (API)

Used by the dashboard and other clients to explore a trained model:

- **Model-level info**: `POST /queries/model-info` — topics, keywords, labels, coherence, and other model-level metrics.
- **Topic-level info**: `POST /queries/topic-info` — detailed info for a single topic.
- **Document–topic weights**: `POST /queries/thetas-by-docs-ids` — thetas for specified document IDs.

All require a valid `model_id` (draft or persisted) and optional `config_path`.

---

## 7. Web dashboard (Version 1)

### 7.1 Authentication and users

- **Sign up / login** (local or via Okta).
- **Sessions** and optional “remember me” behavior.
- **User-scoped data**: Corpora, datasets, and models can be tied to the current user; listing and actions are filtered by ownership where applicable.

### 7.2 Corpus and dataset management

- **Corpora**: Create corpus from uploaded data or existing source; list, view, delete. Associate models with a corpus.
- **Datasets**: Create and manage datasets (e.g. for training or inference).
- **Load data / load corpus** pages for uploading or selecting data.

### 7.3 Training from the UI

- Select corpus (or data source), model type (from registry), and optional training parameters.
- Start training; poll job status and show progress.
- On success, model lives in draft space; user can open the dashboard for that draft or promote it to the persistent store.

### 7.4 Model dashboard (exploration)

- **Topic list**: Topics with labels, keywords, and metrics (e.g. coherence).
- **Top documents per topic**: Representative documents per topic.
- **Similar topics**: For each topic, a list of similar topics (with similarity scores).
- **Document–topic view**: For a selected document, view topic weights (thetas).
- **Topic rename**: Custom labels for topics; stored per model and shown in the UI (`POST /api/models/<model_id>/topics/<topic_id>/rename`, `GET /api/models/<model_id>/topics/renames`).
- **Text info**: For a given document and model, retrieve text and topic-related info (used for detail panels).

### 7.5 Inference from the UI

- Run inference on user-supplied text (or selected data) against a chosen model.
- Results (thetas, etc.) displayed in the UI.

### 7.6 Configuration and LLM (UI)

- **User config**: Per-user overrides (e.g. topic modeling defaults, LLM provider/model for labelling) stored in DB; merged with base `config.yaml` for training and labelling.
- **LLM UI config**: Endpoints to get/set LLM-related settings (e.g. provider, model, Ollama host, API key storage for OpenAI). Used for topic labelling/summarization and for future chat features.

### 7.7 Admin (superuser)

- **Admin page**: List users, toggle admin status, create/delete users.
- **Admin corpora/models**: List and delete corpora or models (e.g. for support or cleanup).
- **Admin stats / audit**: Basic stats and audit log of actions (who did what, when).

---

## 8. Data handling and persistence

### 8.1 Drafts

- **Draft types**: Model (`m_*`), corpus (`c_*`), dataset (`d_*`).
- **Lifecycle**: Training/inference produce drafts; user can inspect them and then **promote** to the persistent store.
- **Promotion**: Triggers indexing (e.g. to Solr and/or Postgres); on success, draft artifacts can be removed. Promoted resources get stable IDs (e.g. `{draft_id}_i`).

### 8.2 API (data)

- **Corpora**: Create, list, get by ID, delete; associate models with a corpus.
- **Models**: List, get by ID, delete; topic renames stored per model.
- **Datasets**: Create and manage dataset resources.
- **Drafts**: Promote draft by ID (model or corpus); returns job ID for the indexing job.

---

## 9. API summary

| Area           | Examples                                                                 |
|----------------|--------------------------------------------------------------------------|
| **Health**     | `GET /health`                                                            |
| **Status**     | `GET /status/jobs/<job_id>`, `DELETE /status/jobs/<job_id>`, SSE logs    |
| **Data**       | Corpora, models, datasets, draft promotion under `/data`                 |
| **Training**   | `POST /train/file`, `POST /train/corpus/<corpus_id>`, TF-IDF endpoints     |
| **Inference**  | `POST /infer/file`, `POST /infer/json`                                  |
| **Queries**    | `POST /queries/model-info`, `topic-info`, `thetas-by-docs-ids`           |
| **Validation** | Endpoints under `/validate` (e.g. config or data checks)                 |

---

## 10. Configuration

- **Main config**: `static/config/config.yaml` — logging, topic modeling (general, traditional, per-model), preprocessing (`tm_preprocessor`), and LLM (providers, API key path, Ollama host, model lists).
- **Model registry**: `static/config/modelRegistry.json` — maps model names to Python class paths (only traditional models in Version 1).
- **User overrides**: Stored in DB; merged with `config.yaml` for the current user when running training or labelling.
- **Environment**: e.g. `.env` for `OPENAI_API_KEY`, `DRAFTS_SAVE`, `API_BASE_URL`, `FLASK_SECRET_KEY`, Okta, etc.

---

## 11. Deployment

- **CLI**: Install package (e.g. `pip install -e .`), run `tova.cli.main` from project root; config and registry paths are relative to the process.
- **Library**: Use `tova` as a dependency; call core dispatchers and model classes directly.
- **Docker**: Makefile-driven `docker compose` — API (port 8000), Web (8080), Solr API (8001), Solr (8983), Postgres (5432), Zookeeper. See main [readme.md](readme.md) for `make up`, `make down`, and service roles.

---

## 12. Out of scope for Version 1

- **LLM-based topic models**: TopicGPT, OpenTopicRAG, or any topic model whose main implementation is LLM-based. Not in the active registry or V1 training/inference flow.
- **Topic-model chatbot**: A dedicated chatbot for querying or interacting with topic models is not part of Version 1. Any existing chat-related endpoints or UI are preparatory and not part of the V1 feature set.

For version rationale, see [VERSION.md](VERSION.md).
