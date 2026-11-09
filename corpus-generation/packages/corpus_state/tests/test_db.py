import sqlite3
import pytest
from pydantic import ValidationError
from corpus_state.types import Character, Relationship, Email, DivergenceRule, Instance, PlotPoint, ResolvedStoryMapping
from corpus_state.db import (
    create_database, get_character, upsert_character, update_character,
    get_relationship, upsert_relationship,
    save_email, get_email, get_email_thread, mark_email_read, get_unread_emails,
    export_blind_sample,
    log_plot_revelation, get_story_progress, upsert_divergence_rule,
    create_instance, get_instance, complete_instance, list_instances,
    bulk_save_emails, export_instance_emails,
    save_story_mapping, get_story_mapping,
)


class TestCorpusIdOnModels:
    def test_character_has_corpus_id_with_default(self):
        c = Character(
            id="mcnulty", wire_character="Jimmy McNulty",
            synthetic_character="John Miller", email_address="j@x.com",
            role="R", organization="O", personality="P", voice_anchor="V",
        )
        assert c.corpus_id == "default"

    def test_character_accepts_explicit_corpus_id(self):
        c = Character(
            id="mcnulty", wire_character="Jimmy McNulty",
            synthetic_character="John Miller", email_address="j@x.com",
            role="R", organization="O", personality="P", voice_anchor="V",
            corpus_id="wire-corp-v1",
        )
        assert c.corpus_id == "wire-corp-v1"

    def test_relationship_has_corpus_id_with_default(self):
        r = Relationship(char_a="c1", char_b="c2", strength=5, status="trusted")
        assert r.corpus_id == "default"

    def test_plot_point_has_corpus_id_with_default(self):
        pp = PlotPoint(id="pager_clone", description="Wiretap established")
        assert pp.corpus_id == "default"

    def test_divergence_rule_has_corpus_id_with_default(self):
        dr = DivergenceRule(
            source_archetype="Mad Scientist",
            synthetic_archetype="Perfectionist",
            rule_description="Channel obsession",
        )
        assert dr.corpus_id == "default"

    def test_instance_has_corpus_id_with_default(self):
        i = Instance(instance_id=0)
        assert i.corpus_id == "default"


class TestCorpusIdSchema:
    def test_characters_table_has_corpus_id_column(self, db):
        cols = {row[1] for row in db.execute("PRAGMA table_info(characters)").fetchall()}
        assert "corpus_id" in cols

    def test_characters_primary_key_is_composite(self, db):
        db.execute(
            "INSERT INTO characters "
            "(corpus_id, id, wire_character, synthetic_character, email_address, role, organization, personality, voice_anchor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("corpus-a", "mcnulty", "Wire", "Synth", "a@x.com", "R", "O", "P", "V"),
        )
        db.execute(
            "INSERT INTO characters "
            "(corpus_id, id, wire_character, synthetic_character, email_address, role, organization, personality, voice_anchor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("corpus-b", "mcnulty", "Wire", "Synth2", "b@x.com", "R", "O", "P", "V"),
        )
        db.commit()
        rows = db.execute("SELECT COUNT(*) FROM characters WHERE id = 'mcnulty'").fetchone()
        assert rows[0] == 2

    def test_relationships_table_has_corpus_id_column(self, db):
        cols = {row[1] for row in db.execute("PRAGMA table_info(relationships)").fetchall()}
        assert "corpus_id" in cols

    def test_plot_points_table_has_corpus_id_column(self, db):
        cols = {row[1] for row in db.execute("PRAGMA table_info(plot_points)").fetchall()}
        assert "corpus_id" in cols

    def test_divergence_map_table_has_corpus_id_column(self, db):
        cols = {row[1] for row in db.execute("PRAGMA table_info(divergence_map)").fetchall()}
        assert "corpus_id" in cols

    def test_instances_table_has_corpus_id_column(self, db):
        cols = {row[1] for row in db.execute("PRAGMA table_info(instances)").fetchall()}
        assert "corpus_id" in cols


@pytest.fixture
def db():
    conn = create_database(":memory:")
    yield conn
    conn.close()


def table_names(db: sqlite3.Connection) -> list[str]:
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r["name"] for r in rows]


class TestSchema:
    def test_creates_all_required_tables(self, db):
        tables = table_names(db)
        assert "characters" in tables
        assert "emails" in tables
        assert "email_reads" in tables
        assert "plot_points" in tables
        assert "divergence_map" in tables
        assert "relationships" in tables
        assert "instances" in tables

    def test_unique_message_id_enforced(self, db):
        db.execute(
            "INSERT INTO emails (sender, recipient, date, message_id, subject, body, folder) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("a@x.com", "b@x.com", "2025-01-01T00:00:00Z", "msg-001", "Hi", "Hello", "\\Inbox"),
        )
        db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO emails (sender, recipient, date, message_id, subject, body, folder) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("c@x.com", "d@x.com", "2025-01-02T00:00:00Z", "msg-001", "Hi", "Dup", "\\Inbox"),
            )

    def test_relationship_strength_check_constraint(self, db):
        for char_id, name in [("c1", "J. Smith"), ("c2", "J. Doe")]:
            db.execute(
                "INSERT INTO characters "
                "(id, wire_character, synthetic_character, email_address, role, organization, personality, voice_anchor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (char_id, "Wire", name, f"{char_id}@corp.com", "Role", "Corp", "trait", "voice"),
            )
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO relationships (char_a, char_b, strength, status) VALUES (?, ?, ?, ?)",
                ("c1", "c2", 11, "trusted"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO relationships (char_a, char_b, strength, status) VALUES (?, ?, ?, ?)",
                ("c1", "c2", 0, "trusted"),
            )
        # valid strength must not raise
        db.execute(
            "INSERT INTO relationships (char_a, char_b, strength, status) VALUES (?, ?, ?, ?)",
            ("c1", "c2", 7, "trusted"),
        )

    def test_relationship_pydantic_rejects_invalid_strength(self, db):
        with pytest.raises(ValidationError):
            Relationship(char_a="c1", char_b="c2", strength=11, status="trusted")
        with pytest.raises(ValidationError):
            Relationship(char_a="c1", char_b="c2", strength=0, status="trusted")


BASE_CHAR = Character(
    id="mcnulty",
    corpus_id="test-corpus",
    wire_character="Jimmy McNulty",
    synthetic_character="John Miller",
    email_address="j.miller@acmecorp.com",
    role="Lead Investigator",
    organization="AcmeCorp",
    personality="tenacious, insubordinate",
    voice_anchor='Terse, sarcastic. Short sentences. Signs off "JM".',
    emotional_state="frustrated",
    blur_rules={"street detective": "corporate investigator"},
)


class TestCharacterQueries:
    def test_upsert_then_get_round_trips_all_fields(self, db):
        upsert_character(db, BASE_CHAR)
        c = get_character(db, "test-corpus", "mcnulty")
        assert c is not None
        assert c.voice_anchor == BASE_CHAR.voice_anchor
        assert c.blur_rules == BASE_CHAR.blur_rules
        assert c.corpus_id == "test-corpus"

    def test_upsert_is_idempotent(self, db):
        upsert_character(db, BASE_CHAR)
        upsert_character(db, BASE_CHAR.model_copy(update={"emotional_state": "resigned"}))
        assert get_character(db, "test-corpus", "mcnulty").emotional_state == "resigned"

    def test_get_unknown_id_returns_none(self, db):
        assert get_character(db, "test-corpus", "nobody") is None

    def test_update_patches_fields_only(self, db):
        upsert_character(db, BASE_CHAR)
        updated = update_character(db, "test-corpus", "mcnulty", {"emotional_state": "hopeful", "role": "Senior Investigator"})
        assert updated.emotional_state == "hopeful"
        assert updated.role == "Senior Investigator"
        assert updated.synthetic_character == "John Miller"  # unchanged

    def test_update_unknown_id_returns_none(self, db):
        assert update_character(db, "test-corpus", "ghost", {"emotional_state": "calm"}) is None

    def test_same_id_different_corpus_are_independent(self, db):
        upsert_character(db, BASE_CHAR)
        other = BASE_CHAR.model_copy(update={"corpus_id": "other-corpus", "role": "Janitor"})
        upsert_character(db, other)
        c1 = get_character(db, "test-corpus", "mcnulty")
        c2 = get_character(db, "other-corpus", "mcnulty")
        assert c1.role == "Lead Investigator"
        assert c2.role == "Janitor"


class TestRelationshipQueries:
    @pytest.fixture(autouse=True)
    def seed_characters(self, db):
        for char_id, name in [("c1", "J. Miller"), ("c2", "J. Doe")]:
            upsert_character(db, Character(
                id=char_id, corpus_id="test-corpus", wire_character="Wire", synthetic_character=name,
                email_address=f"{char_id}@corp.com", role="Role", organization="Corp",
                personality="trait", voice_anchor="voice",
            ))

    def test_upsert_then_get_round_trips(self, db):
        upsert_relationship(db, Relationship(char_a="c1", char_b="c2", strength=7, status="trusted", corpus_id="test-corpus"))
        r = get_relationship(db, "test-corpus", "c1", "c2")
        assert r.strength == 7
        assert r.status == "trusted"

    def test_upsert_overwrites_on_conflict(self, db):
        upsert_relationship(db, Relationship(char_a="c1", char_b="c2", strength=7, status="trusted", corpus_id="test-corpus"))
        upsert_relationship(db, Relationship(char_a="c1", char_b="c2", strength=3, status="strained", corpus_id="test-corpus"))
        assert get_relationship(db, "test-corpus", "c1", "c2").status == "strained"

    def test_get_unknown_pair_returns_none(self, db):
        assert get_relationship(db, "test-corpus", "c1", "c2") is None

    def test_same_pair_different_corpus_are_independent(self, db):
        rel1 = Relationship(char_a="c1", char_b="c2", strength=7, status="trusted", corpus_id="test-corpus")
        rel2 = Relationship(char_a="c1", char_b="c2", strength=3, status="strained", corpus_id="other-corpus")
        upsert_relationship(db, rel1)
        upsert_relationship(db, rel2)
        r1 = get_relationship(db, "test-corpus", "c1", "c2")
        r2 = get_relationship(db, "other-corpus", "c1", "c2")
        assert r1.strength == 7
        assert r2.strength == 3


ROOT_EMAIL = Email(
    sender="c.wilson@acmecorp.com",
    recipient="j.miller@acmecorp.com",
    date="2025-11-10T09:00:00-05:00",
    message_id="<msg-001@acmecorp.com>",
    subject="Q4 review prep",
    body="Can we sync before the meeting?",
    folder="\\Sent",
    parent_mid=None,
)


class TestEmailQueries:
    @pytest.fixture(autouse=True)
    def seed_character(self, db):
        upsert_character(db, BASE_CHAR)  # email_address = j.miller@acmecorp.com

    def test_save_returns_sequential_mids(self, db):
        assert save_email(db, ROOT_EMAIL) == 1
        assert save_email(db, ROOT_EMAIL.model_copy(update={"message_id": "<msg-002@acmecorp.com>"})) == 2

    def test_get_email_returns_none_for_unknown(self, db):
        assert get_email(db, 999) is None

    def test_get_email_round_trips(self, db):
        mid = save_email(db, ROOT_EMAIL)
        e = get_email(db, mid)
        assert e.subject == "Q4 review prep"
        assert e.message_id == "<msg-001@acmecorp.com>"

    def test_thread_of_root_email_has_length_one(self, db):
        mid = save_email(db, ROOT_EMAIL)
        thread = get_email_thread(db, mid)
        assert len(thread) == 1
        assert thread[0].mid == mid

    def test_thread_walks_parent_chain_oldest_first(self, db):
        mid1 = save_email(db, ROOT_EMAIL)
        mid2 = save_email(db, ROOT_EMAIL.model_copy(update={"message_id": "<msg-002@acmecorp.com>", "parent_mid": mid1}))
        mid3 = save_email(db, ROOT_EMAIL.model_copy(update={"message_id": "<msg-003@acmecorp.com>", "parent_mid": mid2}))
        thread = get_email_thread(db, mid3)
        assert len(thread) == 3
        assert thread[0].mid == mid1
        assert thread[2].mid == mid3

    def test_unread_emails_includes_recipient_emails(self, db):
        mid = save_email(db, ROOT_EMAIL)  # recipient = j.miller@acmecorp.com = mcnulty
        assert any(e.mid == mid for e in get_unread_emails(db, "test-corpus", "mcnulty"))

    def test_mark_read_removes_email_from_unread(self, db):
        mid = save_email(db, ROOT_EMAIL)
        mark_email_read(db, mid, "mcnulty")
        assert not any(e.mid == mid for e in get_unread_emails(db, "test-corpus", "mcnulty"))

    def test_export_blind_sample_returns_n_emails(self, db):
        for i in range(1, 6):
            save_email(db, ROOT_EMAIL.model_copy(update={"message_id": f"<msg-00{i}@acmecorp.com>"}))
        sample = export_blind_sample(db, 3)
        assert len(sample) == 3

    def test_export_blind_sample_has_no_character_metadata(self, db):
        save_email(db, ROOT_EMAIL)
        sample = export_blind_sample(db, 1)
        e = sample[0]
        assert not hasattr(e, "wire_character")
        assert not hasattr(e, "voice_anchor")
        assert not hasattr(e, "blur_rules")

    def test_export_blind_sample_caps_at_total_when_n_exceeds_count(self, db):
        save_email(db, ROOT_EMAIL)
        assert len(export_blind_sample(db, 100)) == 1


class TestStoryQueries:
    @pytest.fixture(autouse=True)
    def seed(self, db):
        upsert_character(db, BASE_CHAR)

    def test_log_revelation_creates_plot_point(self, db):
        mid = save_email(db, ROOT_EMAIL)
        log_plot_revelation(db, "test-corpus", "pager_clone", "Wiretap established via pager clone", mid)
        progress = get_story_progress(db, "test-corpus")
        assert len(progress.plot_points) == 1
        assert progress.plot_points[0].id == "pager_clone"
        assert progress.plot_points[0].first_email_mid == mid

    def test_log_revelation_is_idempotent(self, db):
        mid = save_email(db, ROOT_EMAIL)
        log_plot_revelation(db, "test-corpus", "pp1", "First beat", mid)
        log_plot_revelation(db, "test-corpus", "pp1", "First beat", mid)
        assert len(get_story_progress(db, "test-corpus").plot_points) == 1

    def test_upsert_divergence_rule_appears_in_progress(self, db):
        upsert_divergence_rule(db, DivergenceRule(
            corpus_id="test-corpus",
            source_archetype="Mad Scientist",
            synthetic_archetype="Bureaucratic Perfectionist",
            rule_description="Channel obsession into process compliance.",
        ))
        rules = get_story_progress(db, "test-corpus").divergence_rules
        assert len(rules) == 1
        assert rules[0].source_archetype == "Mad Scientist"

    def test_story_progress_isolated_by_corpus(self, db):
        mid = save_email(db, ROOT_EMAIL)
        log_plot_revelation(db, "test-corpus", "pp1", "First beat", mid)
        log_plot_revelation(db, "other-corpus", "pp1", "First beat", mid)
        progress1 = get_story_progress(db, "test-corpus")
        progress2 = get_story_progress(db, "other-corpus")
        assert len(progress1.plot_points) == 1
        assert len(progress2.plot_points) == 1
        assert progress1.plot_points[0].corpus_id == "test-corpus"
        assert progress2.plot_points[0].corpus_id == "other-corpus"


class TestInstanceQueries:
    def test_create_instance_stores_metadata(self, db):
        create_instance(db, "test-corpus", 0, {"story": "jurassic-park"})
        inst = get_instance(db, 0)
        assert inst is not None
        assert inst.status == "in_progress"
        assert inst.metadata == {"story": "jurassic-park"}
        assert inst.corpus_id == "test-corpus"

    def test_create_instance_is_idempotent(self, db):
        create_instance(db, "test-corpus", 0, {"story": "jp"})
        create_instance(db, "test-corpus", 0, {"story": "jp"})
        assert len(list_instances(db)) == 1

    def test_complete_instance_sets_status(self, db):
        create_instance(db, "test-corpus", 0)
        complete_instance(db, 0)
        inst = get_instance(db, 0)
        assert inst.status == "completed"
        assert inst.completed_at is not None

    def test_get_unknown_instance_returns_none(self, db):
        assert get_instance(db, 999) is None

    def test_list_instances_filters_by_status(self, db):
        create_instance(db, "test-corpus", 0)
        create_instance(db, "test-corpus", 1)
        complete_instance(db, 0)
        assert len(list_instances(db, status="completed")) == 1
        assert len(list_instances(db, status="in_progress")) == 1
        assert len(list_instances(db)) == 2

    def test_list_instances_filters_by_corpus_id(self, db):
        create_instance(db, "corpus-a", 0)
        create_instance(db, "corpus-b", 1)
        assert len(list_instances(db, corpus_id="corpus-a")) == 1
        assert len(list_instances(db, corpus_id="corpus-b")) == 1
        assert len(list_instances(db)) == 2

    def test_instances_isolated_by_corpus(self, db):
        create_instance(db, "corpus-a", 0, {"story": "jp"})
        create_instance(db, "corpus-b", 1, {"story": "other"})
        result = list_instances(db, corpus_id="corpus-a")
        assert len(result) == 1
        assert result[0].corpus_id == "corpus-a"


class TestBulkEmailQueries:
    @pytest.fixture(autouse=True)
    def seed(self, db):
        upsert_character(db, BASE_CHAR)
        create_instance(db, "test-corpus", 0)

    def test_bulk_save_inserts_all_emails(self, db):
        emails = [
            Email(
                mid=i+1, sender="a@x.com", recipient="b@x.com",
                date="2025-01-01T00:00:00Z", message_id=f"<msg-{i}@x.com>",
                subject=f"Subject {i}", body=f"Body {i}",
                instance_id=0,
            )
            for i in range(5)
        ]
        result = bulk_save_emails(db, 0, emails)
        assert result["inserted"] == 5
        assert len(result["mids"]) == 5

    def test_bulk_save_uses_explicit_mids(self, db):
        emails = [
            Email(
                mid=1000, sender="a@x.com", recipient="b@x.com",
                date="2025-01-01T00:00:00Z", message_id="<msg-bulk-1@x.com>",
                subject="Test", body="Body", instance_id=0,
            ),
            Email(
                mid=1001, sender="a@x.com", recipient="b@x.com",
                date="2025-01-01T00:00:00Z", message_id="<msg-bulk-2@x.com>",
                subject="Test 2", body="Body 2", instance_id=0,
            ),
        ]
        result = bulk_save_emails(db, 0, emails)
        assert result["mids"] == [1000, 1001]
        assert get_email(db, 1000) is not None
        assert get_email(db, 1001) is not None

    def test_bulk_save_tags_instance_id(self, db):
        emails = [
            Email(
                mid=500, sender="a@x.com", recipient="b@x.com",
                date="2025-01-01T00:00:00Z", message_id="<msg-inst@x.com>",
                subject="Test", body="Body", instance_id=0,
            ),
        ]
        bulk_save_emails(db, 0, emails)
        row = db.execute("SELECT instance_id FROM emails WHERE mid = 500").fetchone()
        assert row["instance_id"] == 0

    def test_bulk_save_rolls_back_on_duplicate_message_id(self, db):
        emails = [
            Email(
                mid=100, sender="a@x.com", recipient="b@x.com",
                date="2025-01-01T00:00:00Z", message_id="<dup@x.com>",
                subject="First", body="Body", instance_id=0,
            ),
            Email(
                mid=101, sender="a@x.com", recipient="b@x.com",
                date="2025-01-01T00:00:00Z", message_id="<dup@x.com>",
                subject="Duplicate", body="Body", instance_id=0,
            ),
        ]
        result = bulk_save_emails(db, 0, emails)
        assert "error" in result
        assert get_email(db, 100) is None  # rolled back

    def test_bulk_save_autoincrement_mid_when_none(self, db):
        emails = [
            Email(
                sender="a@x.com", recipient="b@x.com",
                date="2025-01-01T00:00:00Z", message_id="<msg-auto@x.com>",
                subject="Auto", body="Body", instance_id=0,
            ),
        ]
        result = bulk_save_emails(db, 0, emails)
        assert result["inserted"] == 1
        assert result["mids"][0] is not None
        assert result["mids"][0] > 0


class TestExportQueries:
    @pytest.fixture(autouse=True)
    def seed(self, db):
        upsert_character(db, BASE_CHAR)
        create_instance(db, "test-corpus", 0)
        create_instance(db, "test-corpus", 1)

    def test_export_instance_emails_returns_only_that_instance(self, db):
        bulk_save_emails(db, 0, [
            Email(mid=1, sender="a@x.com", recipient="b@x.com",
                  date="2025-01-01", message_id="<i0-1@x.com>",
                  subject="S", body="B", instance_id=0),
        ])
        bulk_save_emails(db, 1, [
            Email(mid=1000, sender="a@x.com", recipient="b@x.com",
                  date="2025-01-01", message_id="<i1-1@x.com>",
                  subject="S", body="B", instance_id=1),
        ])
        emails = export_instance_emails(db, 0)
        assert len(emails) == 1
        assert emails[0].mid == 1

    def test_export_all_emails(self, db):
        bulk_save_emails(db, 0, [
            Email(mid=1, sender="a@x.com", recipient="b@x.com",
                  date="2025-01-01", message_id="<all-1@x.com>",
                  subject="S", body="B", instance_id=0),
        ])
        bulk_save_emails(db, 1, [
            Email(mid=1000, sender="a@x.com", recipient="b@x.com",
                  date="2025-01-01", message_id="<all-2@x.com>",
                  subject="S", body="B", instance_id=1),
        ])
        emails = export_instance_emails(db, instance_id=None)
        assert len(emails) == 2


class TestStoryMappings:
    def test_save_and_retrieve(self, db):
        mapping = ResolvedStoryMapping(
            corpus_id="test-corpus",
            setting_name="biotech startup",
            setting_organization="NovaBio Inc.",
            setting_domain="genetic research",
            blur_level="medium",
            mode="manual",
            plot_points=[
                {"id": "p1", "description": "First event"},
                {"id": "p2", "description": "Second event"},
            ],
        )
        save_story_mapping(db, mapping)
        loaded = get_story_mapping(db, "test-corpus")
        assert loaded is not None
        assert loaded.corpus_id == "test-corpus"
        assert loaded.setting_name == "biotech startup"
        assert loaded.setting_organization == "NovaBio Inc."
        assert loaded.setting_domain == "genetic research"
        assert loaded.blur_level == "medium"
        assert loaded.mode == "manual"
        assert len(loaded.plot_points) == 2
        assert loaded.plot_points[0]["id"] == "p1"

    def test_get_returns_none_for_missing(self, db):
        assert get_story_mapping(db, "nonexistent") is None

    def test_save_is_idempotent(self, db):
        mapping = ResolvedStoryMapping(
            corpus_id="test-corpus",
            setting_name="biotech",
            setting_organization="X",
            setting_domain="genetics",
            blur_level="light",
            mode="auto",
            plot_points=[{"id": "p1", "description": "d1"}],
        )
        save_story_mapping(db, mapping)
        save_story_mapping(db, mapping)
        loaded = get_story_mapping(db, "test-corpus")
        assert loaded is not None
        assert len(loaded.plot_points) == 1

    def test_rejects_malformed_plot_points(self):
        with pytest.raises(ValidationError):
            ResolvedStoryMapping(
                corpus_id="bad",
                setting_name="x",
                setting_organization="y",
                setting_domain="z",
                blur_level="light",
                mode="manual",
                plot_points=[{"wrong_key": "oops"}],
            )

    def test_rejects_non_string_plot_point_values(self):
        with pytest.raises(ValidationError):
            ResolvedStoryMapping(
                corpus_id="bad",
                setting_name="x",
                setting_organization="y",
                setting_domain="z",
                blur_level="light",
                mode="manual",
                plot_points=[{"id": 42, "description": None}],
            )
