import asyncio
import json
import pytest
from fastmcp import FastMCP
from pydantic import ValidationError

from corpus_state.db import create_database, upsert_character
from corpus_state.tools import email as email_tools
from corpus_state.types import Character, EmailInput


VALID = dict(
    sender="j.miller@acmecorp.com",
    recipient="c.wilson@acmecorp.com",
    date="2025-11-10T09:00:00-05:00",
    message_id="<msg-001@acmecorp.com>",
    subject="Q4 prep",
    body="Can we sync?",
    folder="\\Sent",
    parent_mid=None,
)


class TestEmailInputSchema:
    def test_accepts_valid_email(self):
        e = EmailInput(**VALID)
        assert e.sender == VALID["sender"]

    def test_rejects_missing_sender(self):
        data = {k: v for k, v in VALID.items() if k != "sender"}
        with pytest.raises(ValidationError):
            EmailInput(**data)

    def test_rejects_empty_sender(self):
        with pytest.raises(ValidationError):
            EmailInput(**{**VALID, "sender": ""})

    def test_rejects_empty_message_id(self):
        with pytest.raises(ValidationError):
            EmailInput(**{**VALID, "message_id": ""})

    def test_defaults_folder_to_inbox_when_omitted(self):
        data = {k: v for k, v in VALID.items() if k != "folder"}
        e = EmailInput(**data)
        assert e.folder == "\\Inbox"

    def test_accepts_null_parent_mid_for_root_emails(self):
        e = EmailInput(**{**VALID, "parent_mid": None})
        assert e.parent_mid is None

    def test_accepts_integer_parent_mid_for_replies(self):
        e = EmailInput(**{**VALID, "parent_mid": 3})
        assert e.parent_mid == 3


@pytest.fixture
def tool_db():
    conn = create_database(":memory:", check_same_thread=False)
    upsert_character(conn, Character(
        id="mcnulty",
        corpus_id="test-corpus",
        wire_character="Jimmy McNulty",
        synthetic_character="John Miller",
        email_address="j.miller@acmecorp.com",
        role="Investigator",
        organization="AcmeCorp",
        personality="tenacious",
        voice_anchor='Terse. Signs off "JM".',
    ))
    yield conn
    conn.close()


@pytest.fixture
def mcp_instance(tool_db):
    mcp = FastMCP("test")
    email_tools.register(mcp, tool_db)
    return mcp


def call(mcp: FastMCP, tool: str, **kwargs) -> dict:
    result = asyncio.run(mcp.call_tool(tool, kwargs))
    return json.loads(result.content[0].text)


class TestEmailTools:
    def test_save_email_tool_returns_mid(self, mcp_instance):
        result = call(mcp_instance, "save_email_tool", **VALID)
        assert "mid" in result
        assert isinstance(result["mid"], int)

    def test_save_email_tool_rejects_empty_body(self, mcp_instance):
        result = call(mcp_instance, "save_email_tool", **{**VALID, "body": ""})
        assert "error" in result

    def test_mark_email_read_tool_returns_json(self, mcp_instance):
        mid = call(mcp_instance, "save_email_tool", **VALID)["mid"]
        result = call(mcp_instance, "mark_email_read_tool", email_mid=mid, character_id="mcnulty")
        assert result["ok"] is True
        assert result["mid"] == mid

    def test_get_unread_emails_tool_returns_list(self, mcp_instance):
        call(mcp_instance, "save_email_tool", **{**VALID, "recipient": "j.miller@acmecorp.com"})
        result = call(mcp_instance, "get_unread_emails_tool", corpus_id="test-corpus", character_id="mcnulty")
        assert isinstance(result, list)

    def test_export_blind_sample_tool_returns_list(self, mcp_instance):
        call(mcp_instance, "save_email_tool", **VALID)
        result = call(mcp_instance, "export_blind_sample_tool", n=1)
        assert isinstance(result, list)
        assert len(result) == 1


from corpus_state.tools import instance as instance_tools


@pytest.fixture
def instance_mcp(tool_db):
    mcp = FastMCP("test")
    instance_tools.register(mcp, tool_db)
    return mcp


class TestInstanceTools:
    def test_create_instance_tool(self, instance_mcp):
        result = call(instance_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus", metadata={"story": "jp"})
        assert result["instance_id"] == 0
        assert result["status"] == "in_progress"
        assert result["corpus_id"] == "test-corpus"

    def test_complete_instance_tool(self, instance_mcp):
        call(instance_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus")
        result = call(instance_mcp, "complete_instance_tool", instance_id=0)
        assert result["status"] == "completed"

    def test_list_instances_tool(self, instance_mcp):
        call(instance_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus")
        call(instance_mcp, "create_instance_tool", instance_id=1, corpus_id="test-corpus")
        result = call(instance_mcp, "list_instances_tool")
        assert len(result) == 2

    def test_list_instances_filters_by_corpus(self, instance_mcp):
        call(instance_mcp, "create_instance_tool", instance_id=0, corpus_id="corpus-a")
        call(instance_mcp, "create_instance_tool", instance_id=1, corpus_id="corpus-b")
        result = call(instance_mcp, "list_instances_tool", corpus_id="corpus-a")
        assert len(result) == 1
        assert result[0]["corpus_id"] == "corpus-a"

    def test_reset_instance_tool_deletes_emails(self, instance_mcp, tool_db):
        call(instance_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus")
        # Insert an email directly for this instance
        tool_db.execute(
            "INSERT INTO emails (mid, sender, recipient, date, message_id, subject, body, folder, instance_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (100, "a@x.com", "b@x.com", "2025-01-01", "<reset@x.com>", "S", "B", "\\Inbox", 0),
        )
        tool_db.commit()
        result = call(instance_mcp, "reset_instance_tool", instance_id=0)
        assert result["deleted_emails"] > 0
        # Instance should be back to in_progress
        inst = call(instance_mcp, "list_instances_tool")
        assert inst[0]["status"] == "in_progress"

    def test_complete_nonexistent_instance_returns_error(self, instance_mcp):
        result = call(instance_mcp, "complete_instance_tool", instance_id=999)
        assert "error" in result


@pytest.fixture
def full_mcp(tool_db):
    mcp = FastMCP("test")
    email_tools.register(mcp, tool_db)
    instance_tools.register(mcp, tool_db)
    return mcp


class TestBulkEmailTool:
    def test_bulk_save_returns_mids(self, full_mcp):
        call(full_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus")
        result = call(full_mcp, "bulk_save_emails_tool", instance_id=0, emails=[
            {"mid": 1, "sender": "a@x.com", "recipient": "b@x.com",
             "date": "2025-01-01", "message_id": "<bulk-1@x.com>",
             "subject": "S1", "body": "B1"},
            {"mid": 2, "sender": "a@x.com", "recipient": "b@x.com",
             "date": "2025-01-02", "message_id": "<bulk-2@x.com>",
             "subject": "S2", "body": "B2"},
        ])
        assert result["inserted"] == 2
        assert result["mids"] == [1, 2]

    def test_bulk_save_rejects_empty_list(self, full_mcp):
        call(full_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus")
        result = call(full_mcp, "bulk_save_emails_tool", instance_id=0, emails=[])
        assert "error" in result

    def test_bulk_save_validates_email_fields(self, full_mcp):
        call(full_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus")
        result = call(full_mcp, "bulk_save_emails_tool", instance_id=0, emails=[
            {"sender": "", "recipient": "b@x.com", "date": "2025-01-01",
             "message_id": "<bad@x.com>", "subject": "S", "body": "B"},
        ])
        assert "error" in result


from corpus_state.tools import character as char_tools


@pytest.fixture
def char_mcp(tool_db):
    mcp = FastMCP("test")
    char_tools.register(mcp, tool_db)
    return mcp


class TestCharacterUpsertTool:
    def test_upsert_creates_new_character(self, char_mcp):
        result = call(char_mcp, "upsert_character_tool",
            corpus_id="test-corpus",
            id="daniels",
            wire_character="Cedric Daniels",
            synthetic_character="Charles Davidson",
            email_address="c.davidson@acmecorp.com",
            role="VP Compliance",
            organization="AcmeCorp",
            personality="disciplined, politically astute",
            voice_anchor="Formal tone. Complete sentences.",
        )
        assert result["id"] == "daniels"
        assert result["corpus_id"] == "test-corpus"
        assert result["synthetic_character"] == "Charles Davidson"

    def test_upsert_updates_existing_character(self, char_mcp):
        call(char_mcp, "upsert_character_tool",
            corpus_id="test-corpus",
            id="daniels",
            wire_character="Cedric Daniels",
            synthetic_character="Charles Davidson",
            email_address="c.davidson@acmecorp.com",
            role="VP Compliance",
            organization="AcmeCorp",
            personality="disciplined",
            voice_anchor="Formal.",
        )
        result = call(char_mcp, "upsert_character_tool",
            corpus_id="test-corpus",
            id="daniels",
            wire_character="Cedric Daniels",
            synthetic_character="Charles Davidson",
            email_address="c.davidson@acmecorp.com",
            role="SVP Compliance",
            organization="AcmeCorp",
            personality="disciplined, politically astute",
            voice_anchor="Formal tone.",
        )
        assert result["role"] == "SVP Compliance"


class TestRelationshipTool:
    def test_upsert_relationship_creates_edge(self, char_mcp):
        # Seed two characters first
        for cid, name in [("c1", "J. Miller"), ("c2", "J. Doe")]:
            call(char_mcp, "upsert_character_tool",
                corpus_id="test-corpus",
                id=cid, wire_character="Wire", synthetic_character=name,
                email_address=f"{cid}@corp.com", role="Role", organization="Corp",
                personality="trait", voice_anchor="voice",
            )
        result = call(char_mcp, "upsert_relationship_tool",
            corpus_id="test-corpus",
            char_a="c1", char_b="c2", strength=7, status="trusted",
        )
        assert result["ok"] is True

    def test_upsert_relationship_updates_existing(self, char_mcp):
        for cid, name in [("c1", "J. Miller"), ("c2", "J. Doe")]:
            call(char_mcp, "upsert_character_tool",
                corpus_id="test-corpus",
                id=cid, wire_character="Wire", synthetic_character=name,
                email_address=f"{cid}@corp.com", role="Role", organization="Corp",
                personality="trait", voice_anchor="voice",
            )
        call(char_mcp, "upsert_relationship_tool",
            corpus_id="test-corpus",
            char_a="c1", char_b="c2", strength=7, status="trusted",
        )
        result = call(char_mcp, "upsert_relationship_tool",
            corpus_id="test-corpus",
            char_a="c1", char_b="c2", strength=3, status="strained",
        )
        assert result["ok"] is True


from corpus_state.tools import export as export_tools


@pytest.fixture
def export_mcp(tool_db):
    mcp = FastMCP("test")
    instance_tools.register(mcp, tool_db)
    email_tools.register(mcp, tool_db)
    export_tools.register(mcp, tool_db)
    return mcp


class TestExportTool:
    def test_export_instance_returns_jsonl(self, export_mcp):
        call(export_mcp, "create_instance_tool", instance_id=0, corpus_id="test-corpus")
        call(export_mcp, "bulk_save_emails_tool", instance_id=0, emails=[
            {"mid": 1, "sender": "a@x.com", "recipient": "b@x.com",
             "date": "2025-01-01", "message_id": "<exp-1@x.com>",
             "subject": "S", "body": "B"},
        ])
        result = call(export_mcp, "export_corpus_tool", instance_id=0)
        assert result["count"] == 1
        assert len(result["emails"]) == 1
        assert result["emails"][0]["mid"] == 1

    def test_export_all_returns_all_instances(self, export_mcp):
        for i in range(2):
            call(export_mcp, "create_instance_tool", instance_id=i, corpus_id="test-corpus")
            call(export_mcp, "bulk_save_emails_tool", instance_id=i, emails=[
                {"mid": i * 1000, "sender": "a@x.com", "recipient": "b@x.com",
                 "date": "2025-01-01", "message_id": f"<exp-all-{i}@x.com>",
                 "subject": "S", "body": "B"},
            ])
        result = call(export_mcp, "export_corpus_tool")
        assert result["count"] == 2


class TestEndToEndBatchFlow:
    """Simulates the full batch generation workflow an agent would follow."""

    def test_full_instance_lifecycle(self, tool_db):
        mcp = FastMCP("test")
        char_tools.register(mcp, tool_db)
        email_tools.register(mcp, tool_db)
        instance_tools.register(mcp, tool_db)
        export_tools.register(mcp, tool_db)

        # 1. Bootstrap characters
        call(mcp, "upsert_character_tool",
            corpus_id="e2e-corpus",
            id="mcnulty", wire_character="Jimmy McNulty",
            synthetic_character="John Miller", email_address="j.miller@corp.com",
            role="Investigator", organization="Corp", personality="tenacious",
            voice_anchor="Terse.",
        )
        call(mcp, "upsert_character_tool",
            corpus_id="e2e-corpus",
            id="daniels", wire_character="Cedric Daniels",
            synthetic_character="Charles Davidson", email_address="c.davidson@corp.com",
            role="VP", organization="Corp", personality="disciplined",
            voice_anchor="Formal.",
        )

        # 2. Bootstrap relationship
        call(mcp, "upsert_relationship_tool",
            corpus_id="e2e-corpus",
            char_a="mcnulty", char_b="daniels", strength=6, status="professional",
        )

        # 3. Create instance
        result = call(mcp, "create_instance_tool", instance_id=0, corpus_id="e2e-corpus", metadata={"story": "test"})
        assert result["status"] == "in_progress"

        # 4. Bulk save a wave of emails
        wave_emails = [
            {"mid": 1, "sender": "j.miller@corp.com", "recipient": "c.davidson@corp.com",
             "date": "2025-06-01T09:00:00Z", "message_id": "<e2e-1@corp.com>",
             "subject": "Initial findings", "body": "Sir, I found something."},
            {"mid": 2, "sender": "c.davidson@corp.com", "recipient": "j.miller@corp.com",
             "date": "2025-06-01T10:00:00Z", "message_id": "<e2e-2@corp.com>",
             "subject": "Re: Initial findings", "body": "What did you find?",
             "parent_mid": 1},
        ]
        result = call(mcp, "bulk_save_emails_tool", instance_id=0, emails=wave_emails)
        assert result["inserted"] == 2
        assert result["mids"] == [1, 2]

        # 5. Mark reads
        result = call(mcp, "mark_email_read_tool", email_mid=1, character_id="daniels")
        assert result["ok"] is True

        # 6. Verify unread
        unread = call(mcp, "get_unread_emails_tool", corpus_id="e2e-corpus", character_id="mcnulty")
        mids = [e["mid"] for e in unread]
        assert 2 in mids

        # 7. Complete instance
        result = call(mcp, "complete_instance_tool", instance_id=0)
        assert result["status"] == "completed"
        assert result["num_emails"] == 2

        # 8. Export
        result = call(mcp, "export_corpus_tool", instance_id=0)
        assert result["count"] == 2
        assert result["emails"][0]["sender"] == "j.miller@corp.com"
