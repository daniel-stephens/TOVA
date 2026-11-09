# Corpus Generation Agent

Python pipeline for generating synthetic email corpora grounded in user context.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)

### 2. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Set API credentials

Default config:

- `RCHAT_API_KEY` environment variable (eg, in local `.env`)
- endpoint `https://rchat.nist.gov/proxy`
- model `NVIDIA-Nemotron-3-Super-120B-A12B-FP8`

### 5. Create a run config

```bash
cp config/corpus_config.example.yaml config/corpus_config.local.yaml
```

Edit at least:

- `corpus_id`
- `source_story.path` (directory with one or more `.md` files)
- `story_mapping.synthetic_setting` (name, organization, domain)
- `story_mapping.plot_outline.mode` — `manual` (provide `points`) or `auto` (LLM generates from source story)
- `blur_level` — `light`, `medium`, or `heavy`
- `corpus_size.total_emails` (auto mode) or `corpus_size.num_instances` (manual mode)
- `output.db_path`
- `output.export_path`

### Understanding `story_mapping`

`story_mapping` combines the synthetic context with the plot outline:

```yaml
story_mapping:
  synthetic_setting:
    name: "biotech startup"
    organization: "NovaBio Inc."
    domain: "genetic research"
  plot_outline:
    mode: manual   # or auto
    points:        # required when mode: manual
      - id: inciting_incident
        description: "Handler killed during specimen transfer"
```

**`mode: auto`** — the LLM reads the source story and generates plot points. `corpus_size` sizing fields (`waves_per_instance`, `emails_per_wave`, `emails_per_instance`) are also auto-derived from the plot point count and `blur_level` unless set explicitly.

**`mode: manual`** — you supply the `points` list.

The resolved outline is persisted in the database per `corpus_id`. Reruns and `--resume` reuse it. To regenerate, use a new `corpus_id` or delete the `story_mappings` row in the database.

### Understanding `corpus_size`

`corpus_size` controls generation volume:

- `total_emails`: target total emails across all instances — the primary auto-mode sizing lever; derives `num_instances = ceil(total_emails / emails_per_instance)`
- `num_instances`: independent timelines to generate — required in `manual` plot outline mode; overrides `total_emails` derivation if set explicitly in `auto` mode
- `waves_per_instance`: planning/generation rounds per instance (auto default: 1 per plot point)
- `emails_per_wave`: target emails per round (auto default: 3/4/5 for light/medium/heavy blur)
- `emails_per_instance`: cap on total emails per instance (auto default: waves × epw × 1.5)

In `auto` plot outline mode, set `total_emails` as your single sizing lever — `num_instances` is derived automatically. Any field you set explicitly always takes precedence over auto-derivation.

```yaml
# Auto mode — simplest sizing
corpus_size:
  total_emails: 500       # derives num_instances = ceil(500 / emails_per_instance)
```

```yaml
# Manual mode — num_instances required
corpus_size:
  num_instances: 10
  waves_per_instance: 5
  emails_per_wave: 4
  emails_per_instance: 12
```

Approximate total emails (with explicit values):

```text
num_instances * min(waves_per_instance * emails_per_wave, emails_per_instance)
```

Practical tuning:

- use `total_emails` in auto mode as your top-level budget
- use `emails_per_instance` as a per-timeline hard cap
- use `waves_per_instance` for narrative pacing
- use `emails_per_wave` for per-wave density

### 6. Run the agent

```bash
uv run python -m corpus_agent.main --config config/corpus_config.local.yaml
```

In case of interruptions, resume incomplete runs:

```bash
uv run python -m corpus_agent.main --config config/corpus_config.local.yaml --resume
```

To also extract sensemaking questions after generation:

```bash
uv run python -m corpus_agent.main --config config/corpus_config.local.yaml --extract-questions
```

This runs a post-generation phase that uses the full generation context (plot outline, character map, wave structure, which plot points were revealed in which emails) to produce cross-document sensemaking questions with answer keys and evidence citations. Output is written to `{export_path}/instances/questions_NNNN.json` (per instance) and `{export_path}/questions.json` (combined).

## Examine Generated Corpus

Assuming `output.export_path` is `data/processed/synthetic/<corpus-id>/`:

```bash
# Combined corpus (all instances)
wc -l data/processed/synthetic/<corpus-id>/email_corpus.jsonl
head -1 data/processed/synthetic/<corpus-id>/email_corpus.jsonl | python -m json.tool

# Per-instance files (zero-padded, sortable)
ls data/processed/synthetic/<corpus-id>/instances/
```

Inspect instance status in SQLite:

```bash
sqlite3 data/<db-name>.db "SELECT instance_id, status, num_emails FROM instances ORDER BY instance_id;"
```

## Project Layout

```text
src/corpus_agent/          # pipeline implementation and tests
packages/corpus_state/     # local state package (db/types/tools/resources)
skills/                    # runtime prompt skills (character-mapping/orchestration/email-generation/outline-generation/question-extraction)
config/                    # YAML config templates
```

## Development

Run tests:

```bash
uv run pytest -v
```

If you want to run the optional MCP server interface (not required for `corpus_agent`):

```bash
uv --directory packages/corpus_state run corpus-state-mcp
```
