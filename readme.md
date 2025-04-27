
# TOVA

This project provides a unified interface for training, loading, inferring, and visualizing topic models via both a **command-line interface (CLI)** and a **REST API**. The REST API is designed to serve a frontend.

To avoid code duplication, all core logic (e.g., training, inference) is encapsulated in **commands** — Python functions that can be triggered from both the CLI and the API.


## Architecture Overview

All topic models inherit from the `BaseTMModel` class, which provides shared functionality like config loading, logging, and data handling. From this base, we define:

- `TradTMmodel` — for training traditional topic models (e.g. LDA, CTM)
- `LLMTModel` — for training LLM-based topic models

The system is designed so that any model adhering to the `BaseTMModel` interface can be integrated, regardless of its input/output specifics. Traditional and LLM-based models may require slightly different implementations, but follow the same structure.

To **register** a new model, simply add it to the `MODEL_REGISTRY` in `src/commands/train.py`:

```python
MODEL_REGISTRY = {
    "tomotopy": TomotopyLDATMmodel
}
```

Logging is handled via a centralized logger configured from the YAML config file. Logs are displayed on the console and saved to disk (by default at ``data/logs``).


## Project Structure

> _(To be updated as more models are added)_

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── routers/
│       ├── __init__.py
│       ├── inference.py
│       └── training.py
├── cli.py                       # Typer CLI entry point
├── commands/                    # Shared command logic (used by CLI & API)
│   ├── __init__.py
│   ├── train.py
│   ├── infer.py
│   └── visualize.py
├── preprocessing/
│   ├── __init__.py
│   └── tm_preprocessor.py
├── topic_models/
│   ├── __init__.py
│   ├── base_model.py            # BaseTMModel: config, logging, I/O
│   ├── llm_based/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gptopic_tm_model.py
│   │   └── lloom_tm_model.py
│   └── traditional/
│       ├── __init__.py
│       ├── base.py
│       ├── ctmtm_model.py
│       └── tomotopy_lda_tm_model.py
└── utils/
    ├── __init__.py
    ├── common.py
    └── tm_utils.py
```

## CLI Interface

### Example usage:

```bash
python -m src.cli train run \
  --model tomotopy \
  --data data/bills_sample_1000.csv \
  --text-col tokenized_text \
  --output models/tomotopy
```

## API Interface

Start the API server with:

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

### Example request:

```bash
curl -X POST http://localhost:8000/train/ \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tomotopy",
    "data": "data/bills_sample_100.csv",
    "text_col": "tokenized_text",
    "output": "data/models/tomotopy"
  }'
```