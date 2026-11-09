# corpus-state

Persistent SQLite backend for synthetic email corpus generation. Replaces volatile prompt context with a typed database, preventing narrative drift across long generation runs.

**Stack:** Python 3.11+, Pydantic v2, SQLite

## Python API

Primary usage — import directly into any Python pipeline:

```python
from corpus_state.db import (
    create_database,
    upsert_character, get_character, list_characters,
    upsert_relationship, get_relationship,
    save_email, bulk_save_emails, get_email,
    create_instance, complete_instance, list_instances,
    log_plot_revelation, get_story_progress,
    save_story_mapping, get_story_mapping,
    export_instance_emails, export_all_emails,
)
from corpus_state.types import (
    Character, Relationship, Email, PlotPoint,
    DivergenceRule, StoryProgress, Instance, ResolvedStoryMapping,
)

conn = create_database("data/corpus_state.db")
```

`create_database` runs migrations automatically and returns a `sqlite3.Connection` with row_factory set.

### Key functions

**Characters**

| Function | Description |
|---|---|
| `upsert_character(conn, character)` | Create or replace a character profile |
| `get_character(conn, corpus_id, id)` | Returns `Character \| None` |
| `list_characters(conn, corpus_id)` | Returns `list[Character]` |
| `upsert_relationship(conn, relationship)` | Create or update a relationship |
| `get_relationship(conn, corpus_id, char_a, char_b)` | Returns `Relationship \| None` |

**Emails**

| Function | Description |
|---|---|
| `save_email(conn, email)` | Persist one email, returns assigned `mid` |
| `bulk_save_emails(conn, instance_id, emails)` | Atomic batch insert, rolls back on any failure |
| `get_email(conn, mid)` | Returns `Email \| None` |

**Instances**

| Function | Description |
|---|---|
| `create_instance(conn, instance_id, corpus_id, metadata)` | Register a generation run |
| `complete_instance(conn, instance_id)` | Mark finished, record email count |
| `list_instances(conn, status, corpus_id)` | Filter by status and/or corpus |

**Story**

| Function | Description |
|---|---|
| `log_plot_revelation(conn, corpus_id, plot_point_id, description, email_mid)` | Record a story beat revealed in an email |
| `get_story_progress(conn, corpus_id, instance_id)` | Returns `StoryProgress` |
| `save_story_mapping(conn, mapping)` | Persist resolved plot outline per `corpus_id` (upsert) |
| `get_story_mapping(conn, corpus_id)` | Returns `ResolvedStoryMapping \| None` |

**Export**

| Function | Description |
|---|---|
| `export_instance_emails(conn, instance_id)` | All emails for one instance |
| `export_all_emails(conn, corpus_id)` | All emails across all instances for a corpus |

## Development

```bash
cd packages/corpus_state
uv sync
uv run pytest
```

---

## Optional: MCP server facade

An MCP server wraps the Python API for use with Claude Code or other MCP-compatible hosts. It is not required for the `corpus_agent` pipeline.

```bash
uv --directory packages/corpus_state run corpus-state-mcp
```

### Resources (read-only)

| Resource URI | Returns |
|---|---|
| `character://{corpus_id}/{id}/profile` | Full character profile |
| `relationship://{corpus_id}/{char_a}/{char_b}` | Relationship between two characters |
| `thread://{email_id}/history` | Chronological email chain |
| `story://{corpus_id}/current_progress` | Revealed plot points and divergence rules |
| `instance://{instance_id}/status` | Status and email count |
| `instance://{corpus_id}/summary` | Corpus-level instance counts |

### Tools (state mutations)

**Characters:** `upsert_character_tool`, `update_character_state`, `upsert_relationship_tool`

**Emails:** `save_email_tool`, `bulk_save_emails_tool`, `mark_email_read_tool`, `get_unread_emails_tool`

**Story:** `log_plot_revelation_tool`

**Instances:** `create_instance_tool`, `complete_instance_tool`, `reset_instance_tool`, `list_instances_tool`

**Export:** `export_corpus_tool`, `export_blind_sample_tool`

See each tool's docstring for full signature details.
