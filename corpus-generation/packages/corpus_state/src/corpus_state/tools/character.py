import json
import sqlite3

from fastmcp import FastMCP
from pydantic import ValidationError

from corpus_state.db import update_character, upsert_character, upsert_relationship
from corpus_state.types import Character, Relationship


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.tool()
    def update_character_state(corpus_id: str, id: str, updates: dict) -> str:
        """Update a character's metadata or emotional state.

        Args:
            corpus_id: Corpus this character belongs to (e.g. "wire-corp-v1").
            id: Character ID (e.g. "mcnulty").
            updates: Dict of fields to update. Allowed keys: emotional_state,
                     role, organization, personality, voice_anchor, blur_rules.
        """
        try:
            result = update_character(db, corpus_id, id, updates)
            if result is None:
                return json.dumps({"error": f"Character not found: {corpus_id}/{id}"})
            return json.dumps(result.model_dump(), indent=2)
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def upsert_character_tool(
        corpus_id: str,
        id: str,
        wire_character: str,
        synthetic_character: str,
        email_address: str,
        role: str,
        organization: str,
        personality: str,
        voice_anchor: str,
        emotional_state: str = "neutral",
        blur_rules: dict | None = None,
    ) -> str:
        """Create or fully replace a character profile.

        Use this during initial character mapping to bootstrap all characters
        into the database. For incremental updates to existing characters,
        use update_character_state instead.

        Args:
            corpus_id: Corpus this character belongs to (e.g. "wire-corp-v1").
            id: Character ID slug (e.g. "mcnulty").
            wire_character: Original Wire character name.
            synthetic_character: Mapped character name in synthetic setting.
            email_address: Character's email address.
            role: Character's role in synthetic setting.
            organization: Character's organization.
            personality: Personality description.
            voice_anchor: Writing-style rules to prevent persona drift.
            emotional_state: Current emotional state (default "neutral").
            blur_rules: Dict mapping source traits to synthetic traits.
        """
        try:
            char = Character(
                corpus_id=corpus_id,
                id=id,
                wire_character=wire_character,
                synthetic_character=synthetic_character,
                email_address=email_address,
                role=role,
                organization=organization,
                personality=personality,
                voice_anchor=voice_anchor,
                emotional_state=emotional_state,
                blur_rules=blur_rules or {},
            )
            upsert_character(db, char)
            return json.dumps(char.model_dump(), indent=2)
        except ValidationError as e:
            return json.dumps({"error": str(e)})
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def upsert_relationship_tool(
        corpus_id: str,
        char_a: str,
        char_b: str,
        strength: int,
        status: str,
    ) -> str:
        """Create or update a relationship between two characters.

        Args:
            corpus_id: Corpus these characters belong to.
            char_a: First character ID.
            char_b: Second character ID.
            strength: Relationship strength (1-10).
            status: Relationship status (e.g. "trusted", "adversarial", "dormant").
        """
        try:
            rel = Relationship(corpus_id=corpus_id, char_a=char_a, char_b=char_b, strength=strength, status=status)
            upsert_relationship(db, rel)
            return json.dumps({"ok": True, "corpus_id": corpus_id, "char_a": char_a, "char_b": char_b, "strength": strength, "status": status})
        except ValidationError as e:
            return json.dumps({"error": str(e)})
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})
