import sqlite3
import json
from pathlib import Path

from corpus_state.types import Character, Relationship, Email, PlotPoint, DivergenceRule, StoryProgress, Instance, ResolvedStoryMapping


def create_database(db_path: str | None = None, check_same_thread: bool = False) -> sqlite3.Connection:
    if db_path == ":memory:":
        path = ":memory:"
    else:
        resolved = Path(db_path) if db_path else Path(__file__).parents[4] / "data" / "corpus_state.db"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        path = str(resolved)

    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _run_migrations(conn)
    return conn


def _run_migrations(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                corpus_id           TEXT NOT NULL DEFAULT 'default',
                id                  TEXT NOT NULL,
                wire_character      TEXT NOT NULL,
                synthetic_character TEXT NOT NULL,
                email_address       TEXT NOT NULL,
                role                TEXT NOT NULL,
                organization        TEXT NOT NULL,
                personality         TEXT NOT NULL,
                voice_anchor        TEXT NOT NULL,
                emotional_state     TEXT NOT NULL DEFAULT 'neutral',
                blur_rules          TEXT NOT NULL DEFAULT '{}',
                created_at          TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (corpus_id, id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                mid        INTEGER PRIMARY KEY AUTOINCREMENT,
                sender     TEXT    NOT NULL,
                recipient  TEXT    NOT NULL,
                date       TEXT    NOT NULL,
                message_id TEXT    NOT NULL UNIQUE,
                subject    TEXT    NOT NULL,
                body       TEXT    NOT NULL,
                folder     TEXT    NOT NULL DEFAULT '\\Inbox',
                parent_mid INTEGER REFERENCES emails(mid)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_reads (
                email_mid    INTEGER NOT NULL REFERENCES emails(mid),
                character_id TEXT    NOT NULL,
                read_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (email_mid, character_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plot_points (
                corpus_id       TEXT NOT NULL DEFAULT 'default',
                id              TEXT NOT NULL,
                description     TEXT NOT NULL,
                first_email_mid INTEGER REFERENCES emails(mid),
                revealed_at     TEXT,
                PRIMARY KEY (corpus_id, id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS divergence_map (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                corpus_id           TEXT NOT NULL DEFAULT 'default',
                source_archetype    TEXT NOT NULL,
                synthetic_archetype TEXT NOT NULL,
                rule_description    TEXT NOT NULL,
                created_at          TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                corpus_id TEXT    NOT NULL DEFAULT 'default',
                char_a    TEXT    NOT NULL,
                char_b    TEXT    NOT NULL,
                strength  INTEGER NOT NULL CHECK (strength >= 1 AND strength <= 10),
                status    TEXT    NOT NULL,
                PRIMARY KEY (corpus_id, char_a, char_b)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS instances (
                instance_id  INTEGER PRIMARY KEY,
                corpus_id    TEXT NOT NULL DEFAULT 'default',
                status       TEXT NOT NULL DEFAULT 'in_progress'
                             CHECK (status IN ('pending', 'in_progress', 'completed')),
                num_emails   INTEGER NOT NULL DEFAULT 0,
                metadata     TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS story_mappings (
                corpus_id            TEXT PRIMARY KEY,
                setting_name         TEXT NOT NULL,
                setting_organization TEXT NOT NULL,
                setting_domain       TEXT NOT NULL,
                blur_level           TEXT NOT NULL,
                mode                 TEXT NOT NULL CHECK (mode IN ('auto', 'manual')),
                plot_points          TEXT NOT NULL,
                created_at           TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(emails)").fetchall()
        }
        if "instance_id" not in existing_cols:
            conn.execute("ALTER TABLE emails ADD COLUMN instance_id INTEGER REFERENCES instances(instance_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_emails_instance ON emails(instance_id)")


# ── Character queries ─────────────────────────────────────────────────────────

def get_character(conn: sqlite3.Connection, corpus_id: str, id: str) -> Character | None:
    """Retrieve a character by ID within a corpus. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM characters WHERE corpus_id = ? AND id = ?", (corpus_id, id)
    ).fetchone()
    return Character(**dict(row)) if row else None


def upsert_character(conn: sqlite3.Connection, char: Character) -> None:
    """Insert or replace a character (idempotent)."""
    conn.execute("""
        INSERT INTO characters
            (corpus_id, id, wire_character, synthetic_character, email_address, role, organization,
             personality, voice_anchor, emotional_state, blur_rules)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(corpus_id, id) DO UPDATE SET
            wire_character      = excluded.wire_character,
            synthetic_character = excluded.synthetic_character,
            email_address       = excluded.email_address,
            role                = excluded.role,
            organization        = excluded.organization,
            personality         = excluded.personality,
            voice_anchor        = excluded.voice_anchor,
            emotional_state     = excluded.emotional_state,
            blur_rules          = excluded.blur_rules,
            updated_at          = datetime('now')
    """, (
        char.corpus_id, char.id, char.wire_character, char.synthetic_character, char.email_address,
        char.role, char.organization, char.personality, char.voice_anchor,
        char.emotional_state, json.dumps(char.blur_rules),
    ))
    conn.commit()


def update_character(
    conn: sqlite3.Connection,
    corpus_id: str,
    id: str,
    updates: dict,
) -> Character | None:
    """Patch selected fields on a character. Returns None if not found."""
    existing = get_character(conn, corpus_id, id)
    if existing is None:
        return None
    allowed = {"emotional_state", "role", "organization", "personality", "voice_anchor", "blur_rules"}
    safe = {k: v for k, v in updates.items() if k in allowed}
    merged = existing.model_copy(update=safe)
    upsert_character(conn, merged)
    return get_character(conn, corpus_id, id)


# ── Relationship queries ──────────────────────────────────────────────────────

def get_relationship(
    conn: sqlite3.Connection,
    corpus_id: str,
    char_a: str,
    char_b: str,
) -> Relationship | None:
    """Retrieve a relationship by character pair within a corpus. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM relationships WHERE corpus_id = ? AND char_a = ? AND char_b = ?",
        (corpus_id, char_a, char_b),
    ).fetchone()
    return Relationship(**dict(row)) if row else None


def upsert_relationship(conn: sqlite3.Connection, rel: Relationship) -> None:
    """Insert or replace a relationship (idempotent)."""
    conn.execute("""
        INSERT INTO relationships (corpus_id, char_a, char_b, strength, status)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(corpus_id, char_a, char_b) DO UPDATE SET
            strength = excluded.strength,
            status   = excluded.status
    """, (rel.corpus_id, rel.char_a, rel.char_b, rel.strength, rel.status))
    conn.commit()


# ── Email queries ─────────────────────────────────────────────────────────────

def save_email(conn: sqlite3.Connection, email: Email) -> int:
    """Insert an email and return its message ID (mid)."""
    cursor = conn.execute("""
        INSERT INTO emails (sender, recipient, date, message_id, subject, body, folder, parent_mid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (email.sender, email.recipient, email.date, email.message_id,
          email.subject, email.body, email.folder, email.parent_mid))
    conn.commit()
    return cursor.lastrowid


def get_email(conn: sqlite3.Connection, mid: int) -> Email | None:
    """Retrieve an email by message ID. Returns None if not found."""
    row = conn.execute("SELECT * FROM emails WHERE mid = ?", (mid,)).fetchone()
    return Email(**dict(row)) if row else None


def get_email_thread(conn: sqlite3.Connection, email_mid: int) -> list[Email]:
    """Walk parent_mid chain from email_mid back to root. Returns oldest-first."""
    rows = conn.execute("""
        WITH RECURSIVE thread(mid, sender, recipient, date, message_id, subject, body, folder, parent_mid, instance_id) AS (
            SELECT * FROM emails WHERE mid = ?
            UNION ALL
            SELECT e.* FROM emails e JOIN thread t ON e.mid = t.parent_mid
        )
        SELECT * FROM thread ORDER BY mid ASC
    """, (email_mid,)).fetchall()
    return [Email(**dict(r)) for r in rows]


def mark_email_read(conn: sqlite3.Connection, email_mid: int, character_id: str) -> None:
    """Mark an email as read by a character."""
    conn.execute(
        "INSERT OR IGNORE INTO email_reads (email_mid, character_id) VALUES (?, ?)",
        (email_mid, character_id),
    )
    conn.commit()


def get_unread_emails(conn: sqlite3.Connection, corpus_id: str, character_id: str) -> list[Email]:
    """Return unread emails addressed to character within a corpus that have no read record."""
    rows = conn.execute("""
        SELECT e.* FROM emails e
        JOIN characters c ON (
            e.recipient = c.email_address
            OR e.recipient LIKE c.email_address || ',%'
            OR e.recipient LIKE '%,' || c.email_address || ',%'
            OR e.recipient LIKE '%,' || c.email_address
        )
        WHERE c.corpus_id = ? AND c.id = ?
          AND NOT EXISTS (
              SELECT 1 FROM email_reads r
              WHERE r.email_mid = e.mid AND r.character_id = ?
          )
        ORDER BY e.date ASC
    """, (corpus_id, character_id, character_id)).fetchall()
    return [Email(**dict(r)) for r in rows]


def export_blind_sample(conn: sqlite3.Connection, n: int) -> list[Email]:
    """Return n random emails. Emails contain no character-mapping metadata by schema design."""
    rows = conn.execute("SELECT * FROM emails ORDER BY RANDOM() LIMIT ?", (n,)).fetchall()
    return [Email(**dict(r)) for r in rows]


# ── Story queries ─────────────────────────────────────────────────────────────

def log_plot_revelation(
    conn: sqlite3.Connection,
    corpus_id: str,
    plot_point_id: str,
    description: str,
    email_mid: int,
) -> None:
    """Log a plot revelation (story beat) within a corpus. Idempotent by corpus_id and id."""
    conn.execute("""
        INSERT INTO plot_points (corpus_id, id, description, first_email_mid, revealed_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(corpus_id, id) DO NOTHING
    """, (corpus_id, plot_point_id, description, email_mid))
    conn.commit()


def get_story_progress(conn: sqlite3.Connection, corpus_id: str) -> StoryProgress:
    """Retrieve all plot points and divergence rules in chronological order for a corpus."""
    plot_rows = conn.execute(
        "SELECT * FROM plot_points WHERE corpus_id = ? ORDER BY revealed_at",
        (corpus_id,),
    ).fetchall()
    div_rows = conn.execute(
        "SELECT * FROM divergence_map WHERE corpus_id = ? ORDER BY created_at",
        (corpus_id,),
    ).fetchall()
    return StoryProgress(
        plot_points=[PlotPoint(**dict(r)) for r in plot_rows],
        divergence_rules=[DivergenceRule(**dict(r)) for r in div_rows],
    )


def upsert_divergence_rule(conn: sqlite3.Connection, rule: DivergenceRule) -> None:
    """Insert a divergence rule (source → synthetic mapping)."""
    conn.execute("""
        INSERT INTO divergence_map (corpus_id, source_archetype, synthetic_archetype, rule_description)
        VALUES (?, ?, ?, ?)
    """, (rule.corpus_id, rule.source_archetype, rule.synthetic_archetype, rule.rule_description))
    conn.commit()


def save_story_mapping(conn: sqlite3.Connection, mapping: ResolvedStoryMapping) -> None:
    conn.execute("""
        INSERT INTO story_mappings
            (corpus_id, setting_name, setting_organization, setting_domain,
             blur_level, mode, plot_points)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(corpus_id) DO UPDATE SET
            setting_name         = excluded.setting_name,
            setting_organization = excluded.setting_organization,
            setting_domain       = excluded.setting_domain,
            blur_level           = excluded.blur_level,
            mode                 = excluded.mode,
            plot_points          = excluded.plot_points
    """, (
        mapping.corpus_id, mapping.setting_name, mapping.setting_organization,
        mapping.setting_domain, mapping.blur_level, mapping.mode,
        json.dumps(mapping.plot_points),
    ))
    conn.commit()


def get_story_mapping(conn: sqlite3.Connection, corpus_id: str) -> ResolvedStoryMapping | None:
    row = conn.execute(
        "SELECT * FROM story_mappings WHERE corpus_id = ?", (corpus_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["plot_points"] = json.loads(d["plot_points"])
    return ResolvedStoryMapping(**d)


# ── Instance queries ─────────────────────────────────────────────────────────

def create_instance(
    conn: sqlite3.Connection,
    corpus_id: str,
    instance_id: int,
    metadata: dict | None = None,
) -> Instance:
    conn.execute("""
        INSERT INTO instances (instance_id, corpus_id, status, metadata)
        VALUES (?, ?, 'in_progress', ?)
        ON CONFLICT(instance_id) DO NOTHING
    """, (instance_id, corpus_id, json.dumps(metadata or {})))
    conn.commit()
    return get_instance(conn, instance_id)


def get_instance(conn: sqlite3.Connection, instance_id: int) -> Instance | None:
    row = conn.execute(
        "SELECT * FROM instances WHERE instance_id = ?", (instance_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["metadata"] = json.loads(d["metadata"])
    return Instance(**d)


def complete_instance(conn: sqlite3.Connection, instance_id: int) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM emails WHERE instance_id = ?", (instance_id,)
    ).fetchone()[0]
    conn.execute("""
        UPDATE instances
        SET status = 'completed', num_emails = ?, completed_at = datetime('now')
        WHERE instance_id = ?
    """, (count, instance_id))
    conn.commit()


def list_instances(
    conn: sqlite3.Connection,
    status: str | None = None,
    corpus_id: str | None = None,
) -> list[Instance]:
    query = "SELECT * FROM instances WHERE 1=1"
    params: list = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if corpus_id:
        query += " AND corpus_id = ?"
        params.append(corpus_id)
    query += " ORDER BY instance_id"
    rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        result.append(Instance(**d))
    return result


# ── Bulk operations ──────────────────────────────────────────────────────────

def bulk_save_emails(
    conn: sqlite3.Connection,
    instance_id: int,
    emails: list[Email],
) -> dict:
    mids = []
    try:
        with conn:
            for email in emails:
                mid = email.mid
                conn.execute("""
                    INSERT INTO emails
                        (mid, sender, recipient, date, message_id, subject, body, folder, parent_mid, instance_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mid, email.sender, email.recipient, email.date, email.message_id,
                    email.subject, email.body, email.folder, email.parent_mid, instance_id,
                ))
                mids.append(mid if mid is not None else conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        return {"inserted": len(mids), "mids": mids}
    except sqlite3.Error as e:
        return {"error": str(e), "inserted": 0, "mids": []}


def export_instance_emails(
    conn: sqlite3.Connection,
    instance_id: int | None = None,
) -> list[Email]:
    if instance_id is not None:
        rows = conn.execute(
            "SELECT * FROM emails WHERE instance_id = ? ORDER BY mid", (instance_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM emails ORDER BY mid").fetchall()
    return [Email(**dict(r)) for r in rows]
