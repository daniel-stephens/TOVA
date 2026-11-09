import json
import sqlite3

from fastmcp import FastMCP
from pydantic import ValidationError

from corpus_state.db import (
    save_email, mark_email_read, get_unread_emails, export_blind_sample,
    bulk_save_emails,
)
from corpus_state.types import Email, EmailInput


def register(mcp: FastMCP, db: sqlite3.Connection) -> None:
    @mcp.tool()
    def save_email_tool(
        sender: str,
        recipient: str,
        date: str,
        message_id: str,
        subject: str,
        body: str,
        folder: str = "\\Inbox",
        parent_mid: int | None = None,
    ) -> str:
        """Persist a newly generated email. Validates against schema_email.csv format.

        Args:
            sender: Sender email address.
            recipient: Recipient address(es), comma-separated for multiple.
            date: ISO 8601 timestamp.
            message_id: Unique machine-generated message ID from email header.
            subject: Email subject line.
            body: Email body text.
            folder: Inbox folder (default \\Inbox).
            parent_mid: mid of the email being replied to; None for new threads.
        """
        try:
            validated = EmailInput(
                sender=sender, recipient=recipient, date=date,
                message_id=message_id, subject=subject, body=body,
                folder=folder, parent_mid=parent_mid,
            )
            mid = save_email(db, Email(**validated.model_dump()))
            return json.dumps({"mid": mid})
        except ValidationError as e:
            return json.dumps({"error": str(e)})
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def mark_email_read_tool(email_mid: int, character_id: str) -> str:
        """Mark an email as read by a specific character.

        Args:
            email_mid: The mid of the email.
            character_id: The character ID marking it as read.
        """
        try:
            mark_email_read(db, email_mid, character_id)
            return json.dumps({"ok": True, "mid": email_mid, "character_id": character_id})
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def get_unread_emails_tool(corpus_id: str, character_id: str) -> str:
        """Return emails sent to this character that they have not yet read.

        Args:
            corpus_id: Corpus the character belongs to.
            character_id: The character ID to check.
        """
        try:
            emails = get_unread_emails(db, corpus_id, character_id)
            return json.dumps([e.model_dump() for e in emails], indent=2)
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def export_blind_sample_tool(n: int) -> str:
        """Return n random emails without character mapping metadata, for source leakage testing.

        Args:
            n: Number of emails to sample.
        """
        try:
            sample = export_blind_sample(db, n)
            return json.dumps([e.model_dump() for e in sample], indent=2)
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def bulk_save_emails_tool(instance_id: int, emails: list[dict]) -> str:
        """Save a batch of emails for a generation instance in a single transaction.

        All emails are inserted atomically — if any fails validation or has a
        duplicate message_id, the entire batch is rolled back.

        Args:
            instance_id: The instance these emails belong to.
            emails: List of email dicts. Each must have: sender, recipient, date,
                    message_id, subject, body. Optional: mid (explicit), folder,
                    parent_mid.
        """
        if not emails:
            return json.dumps({"error": "emails list must not be empty"})
        try:
            validated = []
            for raw in emails:
                inp = EmailInput(**raw)
                data = inp.model_dump()
                data["mid"] = raw.get("mid")
                data["instance_id"] = instance_id
                validated.append(Email(**data))
            result = bulk_save_emails(db, instance_id, validated)
            return json.dumps(result)
        except ValidationError as e:
            return json.dumps({"error": str(e)})
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})
