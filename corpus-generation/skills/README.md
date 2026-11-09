# Skills

Markdown prompt skills for synthetic email corpus generation. Each skill is a standalone `.md` file injected as a system prompt — no dependency on any specific runtime or orchestration framework.

## How skills work

A skill is read from disk and passed as the `system` argument to an LLM call. The `corpus_agent` pipeline does this automatically:

```python
skills = load_all_skills()            # loads all SKILL.md files from skills/
system = skills["character-mapping"]  # plain string, ready to inject
```

Any pipeline that can pass a system prompt can use these skills directly.

## Skill reference

### outline-generation

**Purpose:** Read a source story and generate a structured plot outline adapted to a synthetic setting.

**Input context (in the user prompt):**
- Source story text
- Synthetic setting (name, organization, domain)
- Blur level (`light` / `medium` / `heavy`)

**Output:** JSON object with a `"plot_points"` key — array of `{"id": "<snake_case>", "description": "<1-2 sentences>"}` objects, ordered chronologically. 5–12 points.

**Blur level effect:** `light` = close mapping to source; `medium` = adapted beats; `heavy` = only deep narrative skeleton preserved.

---

### character-mapping

**Purpose:** Map source story characters to a synthetic setting and produce structured character profiles and relationships.

**Input context (in the user prompt):**
- Source story text
- Synthetic setting
- Blur level
- Plot outline (so character motivations align with story beats)

**Output:** JSON object with `"characters"` and `"relationships"` arrays. Each character has: `id`, `wire_character`, `synthetic_character`, `email_address`, `role`, `organization`, `personality`, `voice_anchor`, `blur_rules`.

---

### orchestration

**Purpose:** Plan each generation wave — decide which characters send emails, which plot points to reveal, and how many emails to produce.

**Input context (in the user prompt):**
- Current wave number and instance state
- Character roster and relationships
- Plot outline and which beats have been revealed so far
- Wave parameters (emails per wave, emails per instance cap)

**Output:** JSON object with `"emails"` array (planned email stubs) and `"wave_rationale"` string.

---

### email-generation

**Purpose:** Generate the full content of a single email in a specific character's voice.

**Input context (in the user prompt):**
- Sender and recipient character profiles
- Current story context and wave state
- Thread history (if a reply)
- Plot point to reveal (if any)

**Output:** JSON object with: `sender`, `recipient`, `date`, `message_id`, `subject`, `body`, `folder`.

---

## Adding a skill

1. Create `skills/<name>/SKILL.md`
2. Add `"<name>"` to `SKILL_NAMES` in `src/corpus_agent/skills.py`

The pipeline picks it up automatically on next run.

---

## Using with Claude Code

When running corpus generation interactively via Claude Code (rather than the `corpus_agent` pipeline), skills can be invoked as Claude Code slash commands and paired with the `corpus_state` MCP server for state persistence.

In that context:

- **`character-mapping`** uses MCP tools: `upsert_character_tool`, `update_character_state`, `upsert_relationship_tool`
- **`orchestration`** uses MCP tools: `create_instance_tool`, `complete_instance_tool`, `bulk_save_emails_tool`, `mark_email_read_tool`, `get_unread_emails_tool`, `log_plot_revelation_tool`, `export_corpus_tool`, `export_blind_sample_tool`
- **`email-generation`** uses no MCP tools — pure content generation, returns a dict to orchestration
- All three skills read state via `ReadMcpResourceTool` using the resource URIs documented in `packages/corpus_state/README.md`

The `outline-generation` skill has no MCP tool dependencies in either usage mode.
