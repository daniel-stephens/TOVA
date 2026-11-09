---
name: character-mapping
description: Create and manage mappings from The Wire characters to a synthetic story context including adapting their attributes and managing relationships. Use this skill when you need to map The Wire characters to new contexts (e.g., Corporate America), adapt their roles and organizations, manage character metadata and relationships, and support ongoing character development as the synthetic story progresses.
allowed-tools:
  - mcp__corpus-state__upsert_character_tool
  - mcp__corpus-state__update_character_state
  - mcp__corpus-state__upsert_relationship_tool
---

# Character Mapping Skill

This skill creates and manages mappings from The Wire characters to a synthetic story context. It adapts character attributes to fit new settings while preserving character essence, manages relationships and metadata, and supports character development throughout the synthetic story.

## When to Use This Skill

Use this skill when you need to:
- Map The Wire characters to a new context (e.g., Corporate America, organized crime syndicate, political campaign)
- Adapt character roles, organizations, and motivations to fit the new setting
- Manage character metadata including attributes, relationships, and current story state
- Track relationship networks between characters (professional, personal, adversarial, familial)
- Support character evolution as the synthetic story progresses
- Maintain ground-truth traceability to original The Wire characters

## Skill Overview

The character mapping skill uses a template-based approach:
- **Character Profiles**: Standard templates defining attributes (role, company, personality, motivations, etc.)
- **Context Adaptation**: Fill templates based on The Wire characteristics adapted to new context
- **Relationship Management**: Track different relationship types and their evolution
- **State Tracking**: Maintain character state as story progresses

## Character Profile Components

Each mapped character profile includes:

### Core Identity
- **wire_character**: Original The Wire character name
- **synthetic_character**: Mapped character name in new context
- **context_description**: Description of the synthetic setting (e.g., "Fortune 500 technology company")

### Role Adaptation
- **wire_role**: Original role in The Wire (e.g., "homicide detective")
- **synthetic_role**: Adapted role in new context (e.g., "corporate security investigator")
- **role_rationale**: Explanation of how the role translates to new context

### Organization Mapping
- **wire_organization**: Original organization (e.g., "BPD Homicide")
- **synthetic_organization**: Mapped organization (e.g., "GlobalTech Internal Investigations")
- **org_rationale**: Explanation of organizational mapping

### Personality Core
- **core_traits**: Fundamental personality traits that remain consistent
- **adapted_traits**: Traits that may express differently in new context
- **consistency_notes**: How to recognize the same character across contexts

### Motivations & Goals
- **wire_motivations**: Original motivations in The Wire context
- **synthetic_motivations**: Adapted motivations for new context
- **goal_evolution**: How motivations may change as story progresses

### Current State
- **story_position**: Current plot point or story state
- **emotional_state**: Current emotional/mental state
- **recent_developments**: Recent events affecting the character

## Blurring Logic

The `blur_level` determines how closely the synthetic characters mirror the original archetypes:

- **Light**: Preserve 1:1 mapping; tweak minor personality traits to fit the new setting.
- **Medium**: Merge or split minor roles; shift primary motivations to better align with the synthetic context.
- **Heavy**: Significantly alter archetypes (e.g., Mad Scientist -> Bureaucratic Perfectionist); remove 1-2 source characters entirely to reshape the narrative.

## Relationship Management

The skill tracks various relationship types:

### Relationship Categories
- **professional**: Work-related relationships (colleague, supervisor, subordinate)
- **personal**: Family, friendship, romantic relationships
- **adversarial**: Competitive, antagonistic, hostile relationships
- **informational**: Source, mentor, intelligence relationships

### Relationship Attributes
- **strength**: Current strength of relationship (1-10 scale)
- **status**: Current status (active, strained, dormant, etc.)
- **context_notes**: Relationship specifics in current story context
- **evolution_notes**: How relationship has changed over story

## Inputs

The skill expects the following inputs for creating or updating a character mapping:

### For Initial Mapping
- **wire_character_reference**: Reference to The Wire character (from the_wire_s1.md or character bios)
- **target_context**: Description of synthetic story setting (e.g., "Corporate America", "Political Campaign")
	- **blur_level**: Degree of adaptation from original archetypes (light, medium, heavy)
	- **mapping_constraints** (optional): Specific requirements for the mapping
- **current_story_state** (optional): Character's starting point in the synthetic story

### For Updates/Management
- **character_id**: Identifier of the character to update
- **update_type**: What to update (profile, relationship, state, etc.)
- **update_data**: The new information or changes to apply
- **story_progression** (optional): How the story has advanced since last update

## Outputs

The skill produces:

### Character Profile Output
A complete character profile containing all the components listed above, suitable for use in email generation and story development.

### Relationship Output
- Mapping of relationships to other characters
- Relationship attributes and current status
- Notes on relationship evolution and context

### State Tracking Output
- Current character state based on story progression
- Recent developments affecting the character
- Predicted trajectory based on motivations and story direction

## Resource Resolution (Required Before Any Mutation)

`ReadMcpResourceTool` is a built-in Claude Code tool and is always available — it does not need to appear in `allowed-tools`.

Before creating or updating any character or relationship, you MUST read the current state from the MCP server using `ReadMcpResourceTool`. Templated resource URIs do not appear in `ListMcpResourcesTool` — you must construct the URI yourself and call `ReadMcpResourceTool` directly.

| Before calling... | First read this resource URI |
|---|---|
| `upsert_character_tool(corpus_id=C, id=X, ...)` | `character://{C}/{X}/profile` |
| `update_character_state(corpus_id=C, id=X, ...)` | `character://{C}/{X}/profile` — if not found, use `upsert_character_tool` to create first |
| `upsert_relationship_tool(corpus_id=C, char_a=A, char_b=B, ...)` | `relationship://{C}/{A}/{B}` |

- If the resource returns a "not found" error, the entity does not yet exist — proceed with creation.
- If the resource returns data, review the current state before overwriting. Confirm the mutation is intentional.
- **Never skip resource reads.** Even for initial bootstrap, attempt the read first to guard against duplicate creation.

## Mapping Process

> **Before executing any step that calls an MCP tool (e.g., Template Population, Relationship Establishment), complete the Resource Resolution protocol above.**

1. **Reference Analysis**: Examine The Wire character description, role, organization, and key traits
2. **Context Analysis**: Understand the target synthetic setting and its norms/structures
3. **Role Translation**: Determine how the Wire role maps to the new context
4. **Organization Mapping**: Find appropriate organizational equivalent in new context
5. **Personality Preservation**: Identify core traits that must remain consistent
6. **Motivation Adaptation**: Translate Wire motivations to fit new context realities
7. **Template Population**: Fill character profile template with adapted information
8. **Relationship Establishment**: Map existing Wire relationships to new context equivalents
9. **Validation**: Ensure mapping preserves character essence while fitting new context

## Character-Aware Consistency

The skill ensures mapped characters remain recognizable as their Wire counterparts by:
- Preserving core personality traits and decision-making patterns
- Maintaining recognizable speech patterns and communication styles (adapted to context)
- Keeping consistent value systems and moral frameworks (even if expressed differently)
- Preserving fundamental motivations while adapting their expression to new context

## Context Appropriateness

The skill ensures mappings fit the synthetic story setting by:
- Respecting organizational hierarchies and norms of the target context
- Adapting role responsibilities to fit context-appropriate positions
- Translating street-level concepts to context-equivalent concepts (e.g., "the street" → "industry networking events")
- Ensuring character actions and decisions remain plausible in the new setting

## Usage Example

To map Jimmy McNulty to a Corporate America setting:

```
Skill: character-mapping
Inputs:
  wire_character_reference: "Jimmy McNulty: brilliant but highly insubordinate homicide detective who instigates the Barksdale investigation"
  target_context: "Fortune 500 technology company internal investigations division"
  mapping_constraints: "Preserve McNulty's instinctive detective skills and antagonism toward authority, but translate to corporate environment"
  current_story_state: "Beginning of investigation, just assembled makeshift detail"
```

## Integration with Email Generation

This skill works best when used alongside the email generation skill:
1. Use character mapping skill to establish and maintain character profiles
2. Use email generation skill to create emails between mapped characters
3. External workflow uses character profiles to provide context for email generation
4. Character mapping skill updates profiles as story progresses based on email exchanges

## Dependencies

- The Wire character reference data (the_wire_s1.md)
- Target context description
- Character profile templates (can be developed as part of skill usage)
- Relationship tracking mechanisms
- Story progression tracking (for state updates)

## MCP Integration (Required)

The `corpus-state` MCP server is the **mandatory** persistence layer for character state.

### Character Bootstrap Flow

During initial character mapping, use `upsert_character_tool` to create all characters:

```
upsert_character_tool(
    id="mcnulty",
    wire_character="Jimmy McNulty",
    synthetic_character="John Miller",
    email_address="j.miller@acmecorp.com",
    role="Lead Investigator",
    organization="AcmeCorp",
    personality="tenacious, insubordinate",
    voice_anchor='Terse, sarcastic. Short sentences. Signs off "JM".',
    blur_rules={"street detective": "corporate investigator"}
)
```

### Tool Reference

| Operation | MCP Tool |
|-----------|----------|
| Create or replace a character profile | `upsert_character_tool(corpus_id, id, wire_character, ...)` |
| Update emotional state or traits | `update_character_state(corpus_id, id, {emotional_state: "..."})` |
| Look up a character's current state | Read `character://{corpus_id}/{id}/profile` |
| Create/update a relationship | `upsert_relationship_tool(corpus_id, char_a, char_b, strength, status)` |
| Check relationship between characters | Read `relationship://{corpus_id}/{char_a}/{char_b}` |

## Error Handling

- If any MCP tool call fails, **stop and report the error**. Do NOT fall back to writing character data to local `.json`, `.txt`, or `.jsonl` files.
- The `corpus-state` MCP server is the single source of truth for all character and relationship state. File-based persistence is forbidden.
- If the MCP server is unreachable or returns unexpected errors, halt the workflow and notify the user. Do not proceed with stale or assumed data.