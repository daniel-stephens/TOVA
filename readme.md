# TOVA: Topic Visualization & Analysis

TOVA is a topic modeling platform with a plug-in architecture, supporting training and inference via CLI and web interface.

## Table of contents

- [TOVA: Topic Visualization \& Analysis](#tova-topic-visualization--analysis)
  - [Table of contents](#table-of-contents)
  - [Key capabilities](#key-capabilities)
    - [Supported models](#supported-models)
  - [Prerequisites](#prerequisites)
  - [Quick start (Docker)](#quick-start-docker)
  - [Environment configuration (.env)](#environment-configuration-env)
  - [Makefile reference](#makefile-reference)
    - [Building images](#building-images)
    - [Starting and stopping](#starting-and-stopping)
    - [Monitoring](#monitoring)
  - [Services](#services)
  - [Configuration (config.yaml)](#configuration-configyaml)
    - [LLM providers](#llm-providers)
      - [Ollama (local, same machine as Docker)](#ollama-local-same-machine-as-docker)
      - [Ollama on a remote server](#ollama-on-a-remote-server)
  - [CLI usage](#cli-usage)
  - [Library usage](#library-usage)
  - [Troubleshooting](#troubleshooting)
    - [Port already in use](#port-already-in-use)
    - [Postgres authentication error after changing credentials](#postgres-authentication-error-after-changing-credentials)
    - [500 Internal Server Error after rebuild](#500-internal-server-error-after-rebuild)

## Key capabilities

- **Training**
  - Train topic models from the CLI or web UI under a unified pattern
  - Supported models: see Model Support Table for the full list of traditional and LLM-based models
  - Full hyperparameter control over all native parameters of each model
- **Topic Enrichment**
  - Keyword-based topic descriptions out of the box
  - Optionally generate LLM-powered topic labels and summaries
  - Manually refine generated labels
- **Exploration**
  - Interactive dashboard with topic lists, top documents, visualizations, and different evaluation metrics (coherence, entropy, diversity)
  - Suggestions of similar topics based on co-occurrence
- **Inference & Export**
  - Run inference on new inputted documents or a complete corpus
  - Download topic assignments (most representative topic for each document) and document-topic matrices
- **Extensibility**
  - Plug-in architecture to add new topic model classes

### Supported models

| Model | Type | Library |
| --- | --- | --- |
| **TomotopyLDA** | Traditional (Bayesian) | [tomotopy](https://bab2min.github.io/tomotopy/v0.14.0/en/models.html#tomotopy.models.LDAModel) |
| **CTM** | Traditional (Neural) | [contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models) |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24 with the Compose plugin
- `make`
- Git

## Quick start (Docker)

```bash
# 1. Clone the repo
git clone https://github.com/daniel-stephens/TOVA.git && cd TOVA

# 2. Create your environment file (edit ports, credentials, API keys as needed)
cp .env.example .env   # or create it manually (see the next section)

# 3. Build images and start the stack
make smart-up
```

Open the web UI at `http://<host>:<WEB_PORT>` (default `http://localhost:8080`).

> `make smart-up` detects uncommitted changes in `docker/`, `ui/`, and `src/` and
> offers a full `--no-cache` rebuild before starting. Use `make up` to skip the check.

## Environment configuration (.env)

Create a `.env` file in the project root before running any `make` target.
All variables have safe defaults; change only what you need.

```dotenv
##############
# Image tags #
##############
# Keep these in sync with what your team publishes.
VERSION=latest
ASSETS_DATE=latest

##############
# Host ports #
##############
# These control the port exposed on the HOST machine.
# The containers always listen internally on 11000 (api) and 8080 (web).
# Change these if the default ports are already taken on your machine.
API_PORT=11000
WEB_PORT=8080
HOST=0.0.0.0

###############
# Environment #
###############
ENV=development

################
# LLM API keys #
################
# Leave empty to use only local / Ollama models.
OPENAI_API_KEY=

############
# Postgres #
############
# All three variables must be set together and must remain consistent with the
# existing postgres_data volume. If you change them after first run, you must
# wipe the volume first: make reset-db
POSTGRES_USER=tova_user
POSTGRES_PASSWORD=supersecretpassword
POSTGRES_DB=tova_db

###################
# Admin bootstrap #
###################
# Comma-separated e-mails that are always granted admin access on first login.
TOVA_ADMIN_EMAILS=
```

## Makefile reference

### Building images

| Command | What it does |
| --- | --- |
| `make build` | Build all images in order: builder → assets → api → web |
| `make build-api` / `make build-web` | Build a single runtime image (uses cache) |
| `make rebuild-all` | Rebuild everything `--no-cache`, then start the stack |
| `make rebuild-run` | Rebuild only runtime images (api, web) `--no-cache`, then start |
| `make rebuild-api` / `make rebuild-web` | Rebuild a single service `--no-cache` |

### Starting and stopping

| Command | What it does |
| --- | --- |
| `make smart-up` | **Recommended.** Checks for source changes, offers rebuild, then starts |
| `make up` | Build (with cache) and start api, web, postgres |
| `make down` | Stop and remove all containers |
| `make reset-db` | !! Stop containers and wipe the Postgres volume (see [Troubleshooting](#troubleshooting)) |

### Monitoring

| Command | What it does |
| --- | --- |
| `make logs-api` | Stream API logs |
| `make logs-web` | Stream web UI logs |
| `make logs-postgres` | Stream Postgres logs |

## Services

| Service | Internal port | Default host port | Description |
| --- | --- | --- | --- |
| api | 11000 | `API_PORT` (11000) | FastAPI backend |
| web | 8080 | `WEB_PORT` (8080) | Django web UI |
| postgres | 5432 | 5432 | User and session storage |
| solr | 8983 | 8983 | Apache Solr search engine |
| solr-api | 8001 | 8001 | Solr query adapter |
| zookeeper | 2181 | 2181 | Solr coordination |

## Configuration (config.yaml)

Runtime behaviour is controlled by `static/config/config.yaml`. The file has three main sections:

- **`llm`** : provider credentials, hosts, and model allowlists
- **`topic_modeling.general`** : shared defaults (provider, prompt, topic count)
- **Per-model blocks** (`traditional`, `llm_based`, `opentopicrag`, `topicgpt`) : overrides for each model family

### LLM providers

**OpenAI / Azure OpenAI**: set `OPENAI_API_KEY` in `.env`. No changes needed in `config.yaml`.

#### Ollama (local, same machine as Docker)

1. Start Ollama bound to all interfaces so Docker can reach it:

   ```bash
   ollama serve --host 0.0.0.0 --port 11434
   ```

2. In `static/config/config.yaml` use `host.docker.internal` as the hostname:

   ```yaml
   llm:
     ollama:
       host: http://host.docker.internal:11434
       available_models: { ... }

   topic_modeling:
     general:
       llm_provider: "ollama"
       llm_model_type: "gemma3:4b"
       llm_server: "http://host.docker.internal:11434"
   ```

#### Ollama on a remote server

Replace `host.docker.internal` in the former section with the server's IP or hostname, e.g. `http://192.168.1.50:11434`. No docker-compose changes needed.


## CLI usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

```bash
python -m src.tova.cli.main train run \
  --model tomotopyLDA \
  --data data_test/bills_sample_100.csv \
  --text-col tokenized_text \
  --output data/models/tomotopy \
  --tr-params '{"num_topics": 10, "num_iters": 50}'
```

## Library usage

```bash
pip install git+https://github.com/daniel-stephens/TOVA.git@master
# or with UI extras:
pip install "tova[ui] @ git+https://github.com/daniel-stephens/TOVA.git@master"
```

The package exposes modules under `tova.*` for programmatic training and inference.
Build a local wheel with `python -m build --wheel`.

## Troubleshooting

### Port already in use

`make smart-up` / `make up` check ports before building. If a port is taken:

1. Find and stop the conflicting process: `ss -tlnp | grep :<PORT>`
2. Or change `API_PORT` / `WEB_PORT` in `.env` and re-run.

### Postgres authentication error after changing credentials

Postgres does **not** re-initialise an existing volume with new credentials.
If you changed `POSTGRES_USER`, `POSTGRES_PASSWORD`, or `POSTGRES_DB` in `.env`
after the first run, you must wipe the volume:

```bash
make reset-db   # stops containers, removes postgres_data volume
make up         # reinitialises the database with the new credentials
```

> !! `make reset-db` permanently deletes all stored data (users, sessions, model metadata).

### 500 Internal Server Error after rebuild

Usually caused by a stale Postgres volume whose schema or credentials no longer
match the running app. Run `make reset-db` then `make up`.
