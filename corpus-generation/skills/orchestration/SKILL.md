---
name: orchestration
description: Orchestrate the dynamic email generation process for the synthetic corpus by managing email thread state and coordinating workflow and story progression. Use this skill when you need to track which emails have been sent and read, determine when characters should respond, manage the sequence of interactions, and orchestrate the overall synthetic email corpus generation.
allowed-tools:
  - mcp__corpus-state__create_instance_tool
  - mcp__corpus-state__complete_instance_tool
  - mcp__corpus-state__reset_instance_tool
  - mcp__corpus-state__list_instances_tool
  - mcp__corpus-state__bulk_save_emails_tool
  - mcp__corpus-state__save_email_tool
  - mcp__corpus-state__mark_email_read_tool
  - mcp__corpus-state__get_unread_emails_tool
  - mcp__corpus-state__log_plot_revelation_tool
  - mcp__corpus-state__export_corpus_tool
  - mcp__corpus-state__export_blind_sample_tool
---

# Orchestration Skill

This skill orchestrates the dynamic email generation process for the synthetic email corpus project. It manages email thread state (tracking sent/read emails and triggering responses) and coordinates the overall workflow and story progression for the corpus generation.

## When to Use This Skill

Use this skill when you need to:
- Track which emails have been "sent" and are available to be "read" by characters
- Determine when a character should respond to received emails based on story context
- Manage the sequence of character interactions and story advancement
- Orchestrate the overall synthetic email corpus generation process
- Ensure that the corpus meets sensemaking requirements through proper distribution of story information

## Skill Overview

The orchestration skill manages two key aspects of the synthetic corpus generation:

### Email Thread State Management
- Tracking sent emails and their availability for reading
- Determining appropriate response triggers based on character relationships and story context
- Managing email thread history and context for responsive email generation

### Workflow and Story Progression Coordination
- Managing the sequence of character interactions
- Advancing the synthetic story through email exchanges
- Ensuring proper distribution of plot points across the corpus
- Supporting the requirement that answers require cross-document analysis

## Delegation Boundary

This skill is the **lifecycle coordinator**. It:

1. **MANAGES** instance lifecycle (create, complete, reset).
2. **DELEGATES** email content generation to the `email-generation` skill.
3. **PERSISTS** generated emails via `bulk_save_emails_tool` or `save_email_tool`.
4. **TRACKS** read status and story progression via MCP tools.
5. **EXPORTS** the final corpus via `export_corpus_tool`.

It does **NOT**:
- Generate email body text itself. All email content comes from the `email-generation` skill.
- Write to any local file (`.jsonl`, `.json`, `.txt`, or `.csv`). The MCP database is the only persistence target.

### Delegation Flow

1. Before the per-email loop, call `create_instance_tool(instance_id: int, metadata)` to register the generation run. `instance_id` must be an integer. Do this once per run — not per email.

For each email in a wave:

2. Read `character://{corpus_id}/{sender_id}/profile` and `character://{corpus_id}/{recipient_id}/profile` via `ReadMcpResourceTool`.
3. If this is a reply, read `thread://{parent_mid}/history` via `ReadMcpResourceTool` where `parent_mid` is an integer `mid`.
4. Invoke the `email-generation` skill with the character profiles, thread history, and story context.
5. Receive the structured email dict from `email-generation`.
6. Persist via `bulk_save_emails_tool(instance_id, emails)` (preferred) or `save_email_tool(...)` for single emails.
7. Call `mark_email_read_tool(email_mid, character_id)` for each recipient.
8. Call `log_plot_revelation_tool(plot_point_id, description, email_mid)` for any story beats revealed.

## Core Functionality

### 1. Email Thread State Management
- **Sent Email Tracking**: Record when emails are generated and by which character
- **Read Status Management**: Track which emails have been "read" by recipient characters
- **Response Triggering**: Determine when and how characters should respond based on:
  - Character relationships and personalities
  - Story context and current plot points
  - Email content and implications
  - Narrative needs for the synthetic corpus
- **Thread Context Preservation**: Maintain conversation history for generating appropriate responses

### 2. Workflow and Story Progression Coordination
- **Interaction Sequencing**: Determine the order and timing of email exchanges
- **Story Advancement**: Ensure email generation advances the synthetic plot appropriately
- **Information Distribution**: Manage how plot points are distributed across emails to meet sensemaking requirements
- **Character State Updates**: Update character profiles based on email exchanges and story developments
- **Corpus Quality Monitoring**: Verify that the evolving corpus meets the cross-document analysis requirement

## Inputs

The skill expects the following inputs for orchestration:

### For State Management Operations
- **operation_type**: Type of state management operation (track_sent, check_read, trigger_response, etc.)
- **email_data**: Information about the email being tracked (sender, recipient, content, timestamp)
- **character_states**: Current state of relevant characters
- **story_context**: Current plot point or story state
- **relationship_data**: Information about relationships between characters involved
- **blur_level**: Level of noise injection for the corpus (light, medium, heavy)

### For Workflow Coordination Operations
- **coordination_type**: Type of coordination needed (advance_story, sequence_interactions, distribute_info, etc.)
- **current_corpus_state**: Current state of the synthetic email corpus
- **character_profiles**: Profiles of characters involved in the corpus
- **story_objectives**: Goals for story progression and information revelation
- **sensemaking_requirements**: Requirements for cross-document analysis in the corpus

## Outputs

The skill produces:

### State Management Outputs
- **Updated email tracking records**: Sent email logs with read status information
- **Response recommendations**: Suggestions for which characters should respond to which emails and when
- **Thread context for generation**: Context information to pass to email generation skill for responsive emails
- **Read status updates**: Information about which emails have been newly "read"

### Workflow Coordination Outputs
- **Interaction sequence recommendations**: Suggested order and timing for email generation
- **Story progression updates**: Updated story context based on email exchanges
- **Information distribution guidance**: Recommendations for how to distribute plot points across emails
- **Character state updates**: Updated character profiles based on corpus developments
- **Corpus quality assessments**: Evaluations of whether the corpus meets sensemaking requirements

## Resource Resolution (Required Before Each Phase)

`ReadMcpResourceTool` is a built-in Claude Code tool and is always available — it does not need to appear in `allowed-tools`.

Before executing any orchestration phase, you MUST read the relevant MCP resources using `ReadMcpResourceTool`. Templated resource URIs do not appear in `ListMcpResourcesTool` — construct the URI yourself.

### Before starting a generation run:
1. Read `instance://{corpus_id}/summary` — check how many instances exist and their status.
2. Read `story://{corpus_id}/current_progress` — understand which plot points have been revealed.

### Before each wave:
3. Read `instance://{instance_id}/status` — verify the instance is `in_progress`.
4. Call `get_unread_emails_tool(corpus_id, character_id)` for each character who might respond — identify pending response triggers.
   Use the character roster established during character mapping. If the roster is not in context, halt and ask the user to provide the character IDs — the MCP server has no tool to enumerate all characters.

### Before delegating to email-generation:
5. Read `character://{corpus_id}/{sender_id}/profile` and `character://{corpus_id}/{recipient_id}/profile` — pass these to email-generation.
6. If generating a reply, read `thread://{parent_mid}/history` where `parent_mid` is an integer `mid` — pass this to email-generation.

### Before completing an instance:
7. Read `instance://{instance_id}/status` — confirm all waves are persisted before marking complete.

**Never delegate to email-generation without first completing the applicable resource reads.** Pass the full resource data to email-generation so it has ground-truth context.

## Orchestration Process

### Email Thread State Management Process
1. **Input Reception**: Receive email data and character/story context
2. **Sent Email Recording**: Log the email as sent by the sender character
3. **Read Status Evaluation**: Determine read status for recipient characters based on relationships and context
4. **Response Analysis**: Analyze whether and how recipients should respond
5. **Trigger Generation**: Create response triggering recommendations with appropriate context
6. **State Updates**: Update all relevant tracking records

### Workflow Coordination Process
1. **Corpus Analysis**: Analyze current state of synthetic email corpus
2. **Story Evaluation**: Evaluate current story progression against objectives
3. **Interaction Planning**: Plan appropriate character interactions to advance story
4. **Information Planning**: Plan distribution of plot points to meet sensemaking requirements
5. **Sequence Generation**: Generate recommended sequence of email generation operations
6. **Character State Planning**: Plan how character profiles should evolve based on exchanges
7. **Output Generation**: Produce coordination recommendations and updates

## Character-Aware Orchestration

The orchestration ensures character-appropriate behavior by:
- Considering character personalities when determining response likelihood and timing
- Reflecting character relationships in interaction sequencing
- Adapting response triggers based on character motivations and current goals
- Maintaining consistency with established character traits across the corpus

## Story Progression Awareness

The orchestration ensures appropriate story advancement by:
- Tracking revelation of key plot points across the corpus
- Managing pacing of information disclosure
- Ensuring that no single email or cluster contains complete story answers
- Supporting the requirement that answers require cross-document sensemaking
- Advancing the synthetic narrative through logical email exchanges

## Noise Injection

To make the corpus feel like a realistic corporate inbox and break the "too-perfect" narrative of the source story, the orchestration skill supports noise injection:

- **Light**: 5% mundane emails (administrative, routine updates).
- **Medium**: 15% mundane + social emails (watercooler chat, non-work related).
- **Heavy**: 30% mundane + social + parallel noise (unrelated side-threads, corporate fluff).

## Integration with Other Skills

This skill works best when used alongside the other custom skills:
1. **Character Mapping Skill**: Provides character profiles for state management and workflow decisions
2. **Email Generation Skill**: Generates actual emails based on orchestration recommendations
3. **Feedback Loop**: Uses outputs from email generation to update state and inform future orchestration

## Usage Example

To orchestrate a response from Bubbles to Kima Greggs after Kima requests information about a stash house robbery:

```
Skill: orchestration
Inputs:
  operation_type: "trigger_response"
  email_data: {
    sender: "kima_greggs_narcotics",
    recipient: "bubbles_informant",
    subject: "Request for information on stash house robbery",
    body: "Bubbles, I need to know what you saw at the Franklin Terrace robbery last night. This is important.",
    timestamp: "2026-04-15T14:30:00Z"
  }
  character_states: {
    kima_greggs_narcotics: {current_focus: "investigation", relationship_to_bubbles: "trusted_informant"},
    bubbles_informant: {current_motivation: "self_preservation", relationship_to_kima: "information_source"}
  }
  story_context: "Kima is investigating a stash house robbery that may connect to Omar Little's activities"
  relationship_data: {
    relationship_type: "professional_informant",
    trust_level: "medium",
    interaction_history: "limited_but_productive"
  }
```

## Dependencies

- Character profiles (from character mapping skill or direct definition)
- Email schema definition (`schema_email.csv`)
- Story context and plot point information
- Relationship data between characters
- Current state of the synthetic email corpus
- Sensemaking requirements for the corpus

## MCP Integration (Required)

The `corpus-state` MCP server is the **mandatory** persistence layer. Do not write to files directly.

### Instance-Aware Workflow

Before generating any emails, create an instance:

1. Call `create_instance_tool(instance_id, corpus_id, metadata)` for the current generation run
2. For each wave in the plot outline:
   - Call `get_unread_emails_tool(corpus_id, character_id)` to identify pending response triggers
   - Generate reply emails via the email-generation skill
   - Call `bulk_save_emails_tool(instance_id, emails)` to persist the entire wave
   - Call `mark_email_read_tool(email_mid, character_id)` for each recipient
   - Call `log_plot_revelation_tool(corpus_id, plot_point_id, description, email_mid)` for story beats
3. Call `complete_instance_tool(instance_id)` when done
4. Check `instance://{corpus_id}/summary` to track overall progress across instances

### Tool Reference

| Operation | MCP Tool |
|-----------|----------|
| Start a generation run | `create_instance_tool(instance_id, corpus_id, metadata)` |
| Check unread emails for a character | `get_unread_emails_tool(corpus_id, character_id)` |
| Save a batch of emails | `bulk_save_emails_tool(instance_id, emails)` |
| Mark an email as read | `mark_email_read_tool(email_mid, character_id)` |
| Record a story beat | `log_plot_revelation_tool(corpus_id, plot_point_id, description, email_mid)` |
| Check story progress | Read `story://{corpus_id}/current_progress` |
| Mark instance done | `complete_instance_tool(instance_id)` |
| Re-generate an instance | `reset_instance_tool(instance_id)` |
| Export final corpus | `export_corpus_tool(instance_id)` |
| Export blind sample for leakage testing | `export_blind_sample_tool(n)` |

## Error Handling

- If any MCP tool call fails, **stop and report the error**. Do NOT fall back to writing emails, character data, or corpus state to local `.jsonl`, `.json`, `.txt`, or `.csv` files.
- The `corpus-state` MCP server is the single source of truth for all corpus data. File-based persistence is forbidden.
- If `bulk_save_emails_tool` fails (e.g., duplicate `message_id`, validation error), the entire batch is rolled back. Diagnose the specific validation failure, fix the offending email(s), and retry the batch. Do not split the batch into individual `save_email_tool` calls as a workaround — fix the root cause.
- If the MCP server is unreachable, halt the workflow and notify the user. Do not proceed with stale data or assumed state.
- If `email-generation` returns malformed data (missing required fields, empty body), reject it and re-invoke with corrected inputs. Do not persist incomplete emails.