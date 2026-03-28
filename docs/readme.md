
# TOVA

This project provides a unified interface for training, loading, inferring, and visualizing topic models via both a **command-line interface (CLI)** and a **REST API**. The REST API is designed to serve a frontend.

To avoid code duplication, all core logic (e.g., training, inference) is encapsulated in **core** — Python functions that can be triggered from both the CLI and the API.


## Architecture Overview

All topic models inherit from the `BaseTMModel` class, which provides shared functionality like config loading, logging, and data handling. From this base, we define:

- `TradTMmodel` — for training traditional topic models (e.g. LDA, CTM)
- `LLMTModel` — for training LLM-based topic models

The system is designed so that any model adhering to the `BaseTMModel` interface can be integrated, regardless of its input/output specifics. Traditional and LLM-based models may require slightly different implementations, but follow the same structure.

To **register** a new model, simply add it to the `MODEL_REGISTRY` in `static/config/modelRegistry.json`:

```python
MODEL_REGISTRY = {
        "tomotopyLDA": "src.topic_models.models.traditional.tomotopy_lda_tm_model.TomotopyLDATMmodel"
}
```
> Note that full path to the module needs to be provided.

Logging is handled via a centralized logger configured from the YAML config file. Logs are displayed on the console and saved to disk (by default at ``data/logs``).

### Note on automatic topic labelling for traditional topic models: 
If labelling is set to True and an LLM model is specified (it can also be passed through kwargs),
```yaml
################################
# TOPIC MODELING CONFIGURATION #
################################
topic_modeling:
  traditional:
    ...
    do_labeller: True
    labeller_model_type: "qwen:32b"
    ...
  ```
  the Prompter class is used to generate a label for the topic based on the topic keywords and most representative documents. For doing so, it can either do it based on a closed-source model (gpt-like) for which a `.env` file with the OpenAI APIKEY should be configured from the prompter params:
  ```yaml
  ################################
  #          PROMPTER            #
  ################################
  llm:
      ...
      path_api_key: .env
      ...
  ```
  otherwise, Ollama models can be used. For this, the url where the ollama server is deployed should be updated:
  ```yaml
  ################################
  #          PROMPTER            #
  ################################
  llm:
    ...
    ollama:
      ...
      host: http://kumo01.tsc.uc3m.es:11434
  ```


## Project Structure

> _(To be updated as more models are added)_

```
src/
├── api/                          # FastAPI backend
│   ├── docs/                     # Examples for API docs
│   │   ├── examples.py
│   │   └── __init__.py
│   ├── main.py                   # API entrypoint — run with Uvicorn
│   ├── models/                   # Request/response models
│   │   ├── data_schemas.py       # Shared data records used across API schemas
│   │   ├── infer_requests.py     # Inference-specific request models
│   │   ├── query_requests.py     # Query-specific request models
│   │   └── train_requests.py     # Training-specific request models
│   └── routers/                  # API route definitions
│       ├── inference.py
│       ├── queries.py
│       └── training.py
│
├── cli/                          # Typer-based CLI
│   ├── main.py                   # CLI entrypoint
│   └── commands/                 # CLI commands
│       ├── train.py              # "" for training
│       ├── infer.py              # "" for inference
│       └── model_query.py        # "" for querying trained models
│
├── core/                         # Shared dispatch logic between CLI/API
│   └── dispatchers.py            
│
├── preprocessing/                # Text cleaning, tokenization, etc. (NOT IMPLEMENTED YET)
│   └── tm_preprocessor.py
│
├── prompter/                     # Prompt-generation utilities (for LLM-based models)
│   ├── prompter.py
│   └── prompts/
│       └── labelling_dft.txt     # Labelling prompt
│
├── topic_models/                 # All topic modeling logic (traditional & LLM-based)
│   ├── models/                   # Model implementations
│   │   ├── base_model.py         # Abstract base class for all topic models
│   │   ├── llm_based/            # Topic models using LLMs
│   │   │   ├── base.py
│   │   │   ├── gptopic_tm_model.py
│   │   │   └── lloom_tm_model.py
│   │   └── traditional/          # Classical models
│   │       ├── base.py
│   │       ├── ctmtm_model.py
│   │       └── tomotopy_lda_tm_model.py
│   ├── tm_manager.py             # Class for managing models (NEEDS RESTRUCTURING)
│   └── tm_model.py               # Generic representation of topic models
│
├── utils/                        # General-purpose utilities
│   ├── common.py                 # Logging, config, registry loaders
│   └── tm_utils.py               # Data preparation for topic models
│
└── static/                       # Config files (model registry, config YAML, etc.)
```

## CLI Interface

### Example usage:

#### TRAIN

```bash
python -m src.cli.main train run \
  --model tomotopyLDA \
  --data data/bills_sample_100.csv \
  --text-col tokenized_text \
  --output data/models/tomotopy
```

#### INFERENCE

```bash
python -m src.cli.main infer run --data data/bills_sample_100.csv   --text-col tokenized_text   --model-path data/models/tomotopy
```

#### GET MODEL INFO

```bash
python -m src.cli.main model_query model-info --model-path data/models/tomotopy
```

## API Interface

Start the API server with:

```bash
python -m uvicorn src.api.main:app --reload --port 8989
```

### Example request:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8989/train/file' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "config_path": "static/config/config.yaml",
  "data_path": "data/bills_sample_100.csv",
  "do_preprocess": false,
  "id_col": "id",
  "model": "tomotopyLDA",
  "output": "data/models/50_tpcs",
  "text_col": "tokenized_text",
  "training_params": {
    "alpha": 0.05,
    "num_topics": 50
  }
}'
```