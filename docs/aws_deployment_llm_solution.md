# How aws_deployment Solved the LLM / Config vs Request Params Issue

## When it was like this

- The UI (and saved user LLM config) could send `llm_provider`, `llm_model_type`, `llm_server`, `llm_api_key` in training params.
- Traditional models (LDA, CTM) received these via `**tr_params` in the dispatcher. Their `__init__` did not accept or expect these keys, so training could fail or behave oddly when the UI sent LLM settings.

## How it was solved on aws_deployment

**Commit: `1411394` — "training fixed" (Thu Feb 19, 2026)**

### 1. Dispatcher: strip LLM-only params for traditional models

**File: `src/tova/core/dispatchers.py`**

- Import `TradTMmodel`.
- Define `_LLM_ONLY_TR_PARAMS = frozenset({"llm_provider", "llm_model_type", "llm_server", "llm_api_key"})`.
- Before calling the model constructor, if the model class is a traditional model (`issubclass(model_cls, TradTMmodel)`), filter out those keys from `tr_params` so they are never passed to the topic model.
- Result: traditional model training always uses **config only** for LLM (labeller/summarizer); UI/saved LLM config is not passed through.

### 2. TradTMmodel: accept and ignore extra kwargs

**File: `src/tova/topic_models/models/traditional/base.py`**

- Add `**kwargs` to `TradTMmodel.__init__`.
- Document that extra kwargs (e.g. from the UI) are accepted and ignored; traditional models read LLM settings from config in `BaseTMModel`.

So even if any LLM params slip through, the base does not break.

## Making Ollama work on aws_deployment

Ollama for labeller/summarizer was made to work by **config**, not by request params:

- **`static/config/config.yaml`** (e.g. in commit `8ca1430` "ollama in progress"):
  - `topic_modeling.general.llm_provider: "ollama"`
  - `topic_modeling.general.llm_model_type: "gemma3:4b"`
  - `topic_modeling.general.llm_server: "http://host.docker.internal:11434"`
  - `llm.ollama.host` set as needed (e.g. for Docker).

With the "training fixed" logic, the UI/saved LLM settings are **not** sent to traditional models, so the labeller/summarizer use only these config values. To use Ollama you set the above in `config.yaml` (and optionally in the `llm` section for the Prompter).

## Summary

| Approach | Request params (UI/saved) | Config |
|----------|---------------------------|--------|
| **aws_deployment "training fixed"** | Stripped for traditional models; not used. | Only source for traditional LDA/CTM labeller/summarizer. |
| **To use Ollama** | N/A for traditional. | Set `llm_provider`, `llm_model_type`, `llm_server` (and `llm.ollama.host`) in `config.yaml`. |
