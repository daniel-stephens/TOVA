---
name: email-generation
description: Generate synthetic emails for character-based corpus where emails are created dynamically based on character interactions and story progression. Use this skill when you need to create email content that reflects specific character voices, story context, and supports dynamic email threads where characters can read and respond to each other's emails.
allowed-tools: []
---

# Email Generation Skill

This skill generates synthetic emails for the character-based corpus project. It creates emails that reflect specific character voices, story context, and supports dynamic email threads where characters can read and respond to each other's emails.

## When to Use This Skill

Use this skill when you need to:
- Generate email content that reflects specific character personalities and motivations
- Create emails that reference current story events and plot points
- Structure emails according to the defined email schema
- Generate responsive emails where one character replies to another's email
- Create a corpus of emails that requires cross-document sensemaking to understand the full story

## Skill Overview

The email generation skill uses a hybrid approach:
- **Email Structure**: Templates for headers and metadata (Message-ID, date formatting, etc.)
- **Content Generation**: LLM to generate email body based on character profiles and story context
- **Dynamic Handling**: Focuses on generating individual emails on request; external workflow manages email thread state and triggering responses

## Delegation Boundary

This skill is a **pure content generator**. It:

1. **READS** MCP resources (character profiles, thread history, story progress) for context.
2. **GENERATES** email content (subject, body, metadata) as structured data.
3. **RETURNS** the generated email as a dict/JSON object to the caller.

It does **NOT**:
- Call `save_email_tool` or `bulk_save_emails_tool` to persist emails.
- Call `mark_email_read_tool` to update read status.
- Call `log_plot_revelation_tool` to record story beats.
- Write to any local file (`.jsonl`, `.json`, `.txt`, or `.csv`).

**Persistence is the orchestration skill's responsibility.** When this skill is invoked by orchestration, return the structured email data and let the orchestrator handle database commits.

## Inputs

The skill expects the following inputs:
- **sender_character**: ID or profile of the character sending the email
- **recipient_character**: ID or profile of the character receiving the email  
- **story_context**: Current plot point or story state
- **parent_email_id** (optional): ID of the email being replied to (for thread generation)
- **email_intent** (optional): Specific purpose or intent for the email (e.g., "share information", "confront", "report progress")

## Outputs

The skill produces a fully formatted email matching the schema defined in `data/raw/synthetic-story/the-wire/schema_email.csv`:
- **mid**: Assigned by the database on insert — do NOT include it in the returned dict
- **parent_mid** (optional): The integer `mid` of the email being replied to. Include this in the returned dict when generating a reply — it is caller-supplied, not database-assigned.
- **sender**: Email address of the sender
- **recipient**: Email address(es) of the recipient(s), comma-separated for multiple recipients
- **date**: Timestamp of when the email was sent
- **message_id**: Original, machine-generated alphanumeric Message-ID from the email header
- **subject**: Email subject line
- **body**: Actual text content/body of the email
- **folder**: Specific inbox folder the email was extracted from (e.g., \Inbox, \Sent)

## Resource Resolution (Required Before Generating)

`ReadMcpResourceTool` is a built-in Claude Code tool and is always available — it does not need to appear in `allowed-tools`.

Before generating any email content, you MUST read the relevant MCP resources using `ReadMcpResourceTool`. Templated resource URIs do not appear in `ListMcpResourcesTool` — construct the URI yourself.

### For every email:
1. Read `character://{corpus_id}/{sender_id}/profile` — extract voice_anchor, personality, emotional_state, and role.
2. Read `character://{corpus_id}/{recipient_id}/profile` — understand the recipient's role and relationship to sender.

### For reply emails (when `parent_mid` is provided):
3. Read `thread://{parent_mid}/history` where `parent_mid` is an integer `mid` — the value returned when the parent email was saved. Include the entire chain in your generation context.

### For emails that reveal story beats:
4. Read `story://{corpus_id}/current_progress` — check which plot points have already been revealed to avoid duplication.

**Never generate email content without first completing the applicable resource reads above.** The resources provide the ground-truth context that ensures character voice consistency and narrative coherence.

## Generation Process

1. **Context Preparation**: Gather sender profile, recipient profile, and current story context
2. **Structure Creation**: Use template to establish email headers and metadata format
3. **Content Generation**: Use LLM to generate email body based on:
   - Sender character traits and current motivations
   - Recipient character profile and relationship to sender
   - Story context and recent events
   - Email thread context (if responding to previous email)
   - Specific email intent/purpose
4. **Schema Compliance**: Format the generated content to match the required email schema
5. **Output**: Return the complete, structured email

## Character-Aware Generation

The skill ensures emails sound authentic to specific characters by:
- Incorporating character-specific language patterns and tone
- Reflecting character motivations and current emotional state
- Referencing character-appropriate knowledge and concerns
- Maintaining consistency with established character traits across multiple emails

## Context Awareness

Generated emails appropriately reference:
- Current plot points and story developments
- Recent events in the synthetic story
- Character relationships and interactions
- Organizational context and role-specific concerns

## Thread Awareness (when generating responses)

When creating a response email:
- References or acknowledges the parent email appropriately
- Maintains consistent tone and topic progression
- Shows appropriate character reaction to received information
- Follows logical conversation flow

## Usage Example

To generate an email from Jimmy McNulty (as a corporate investigator) to Cedric Daniels (as internal affairs) about a recent development in the investigation:

```
Skill: email-generation
Inputs:
  sender_character: jimmy_mcnulty_corporate_investigator
  recipient_character: cedric_daniels_internal_affairs
  story_context: "The detail has just cloned the Barksdale pager system and is beginning to monitor communications"
  email_intent: "share preliminary findings and request resources"
```

## Integration with Character Mapping

This skill works best when used alongside the character mapping skill:
1. Use character mapping skill to establish and maintain character profiles
2. Use email generation skill to create emails between mapped characters
3. External workflow tracks email thread state and triggers responsive email generation

## Dependencies

- Access to character profiles (from character mapping or direct definition)
- Story context/current plot point information
- LLM for content generation (handled by the hosting system)
- Email schema definition (`schema_email.csv`)

## MCP Integration (Read-Only)

This skill reads from the `corpus-state` MCP server but does **not** write to it.

### Resources Used

| Purpose | Resource URI |
|---------|-------------|
| Sender voice and personality | `character://{corpus_id}/{sender_id}/profile` |
| Recipient context | `character://{corpus_id}/{recipient_id}/profile` |
| Thread history for replies | `thread://{parent_mid}/history` |
| Story progress (avoid duplicate reveals) | `story://{corpus_id}/current_progress` |

### Output Format

Return generated emails as structured dicts matching this schema:

```python
{
    "sender": "j.miller@acmecorp.com",
    "recipient": "c.daniels@acmecorp.com",
    "date": "2026-05-26T09:15:00Z",
    "message_id": "<uuid@acmecorp.com>",
    "subject": "Re: Q2 Compliance Review",
    "body": "Cedric,\n\nI pulled the access logs...",
    "folder": "\\Sent",
    "parent_mid": 1001
}
```

The orchestration skill receives this output and calls `save_email_tool` or `bulk_save_emails_tool` to persist it.

## Error Handling

- If an MCP resource read fails (e.g., character profile not found), **stop and report the error**. Do NOT generate email content with assumed or fabricated character data.
- Do NOT fall back to writing email data to local `.jsonl`, `.txt`, or `.json` files. This skill does not persist — it returns data to the caller.
- If the MCP server is unreachable, halt and notify the user.