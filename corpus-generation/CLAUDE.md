# CLAUDE.md

Guidance for coding agents working in this repository.

## Scope

This repository contains the standalone corpus generation stack:

- `src/corpus_agent/` — async corpus generation pipeline
- `packages/corpus_state/` — local state package (SQLite + optional MCP server facade)
- `skills/` — skill markdown injected into prompts at runtime

## Runtime Assumptions

- Main entry point: `uv run python -m corpus_agent.main --config <yaml>`
- `corpus_agent` is standalone Python and uses `corpus_state` directly.
- Do not assume Claude Code runtime or a running MCP server for normal execution.

## Environment

- Use `uv` for environment/package management.
- Typical setup:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Testing

- Full tests:

```bash
uv run pytest -v
```

- `corpus_state` tests only:

```bash
uv run pytest packages/corpus_state/tests -v
```

## LLM Defaults

`config/corpus_config.example.yaml` defaults to:

- env var: `RCHAT_API_KEY`
- endpoint: `https://rchat.nist.gov/proxy`
- model: `NVIDIA-Nemotron-3-Super-120B-A12B-FP8`

## Repository Boundaries

- Keep generated corpora and scratch outputs out of git.
- Treat `skills/` as runtime assets; changing them changes model behavior.
- Keep `corpus_state` import path stable (`import corpus_state...`).
