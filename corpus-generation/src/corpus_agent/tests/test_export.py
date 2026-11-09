import json

from corpus_state.db import bulk_save_emails, complete_instance, create_database, create_instance
from corpus_state.types import Email

from corpus_agent.phases.export import export_corpus


class TestExportCorpus:
    def test_exports_to_jsonl(self, tmp_path):
        conn = create_database(":memory:")
        create_instance(conn, "test", 1)
        bulk_save_emails(
            conn,
            1,
            [
                Email(
                    sender="a@b.com",
                    recipient="c@d.com",
                    date="2026-01-01",
                    message_id="<1@t>",
                    subject="Hi",
                    body="Hello",
                ),
                Email(
                    sender="c@d.com",
                    recipient="a@b.com",
                    date="2026-01-02",
                    message_id="<2@t>",
                    subject="Re: Hi",
                    body="Hey",
                ),
            ],
        )
        complete_instance(conn, 1)

        export_path = tmp_path / "output"
        export_corpus(conn, str(export_path), corpus_id="test")

        corpus_file = export_path / "email_corpus.jsonl"
        assert corpus_file.exists()
        lines = corpus_file.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["sender"] == "a@b.com"
        assert "mid" in first

    def test_exports_per_instance(self, tmp_path):
        conn = create_database(":memory:")
        create_instance(conn, "test", 1)
        bulk_save_emails(
            conn,
            1,
            [
                Email(
                    sender="a@b.com",
                    recipient="c@d.com",
                    date="2026-01-01",
                    message_id="<1@t>",
                    subject="Hi",
                    body="Hello",
                ),
            ],
        )
        complete_instance(conn, 1)

        create_instance(conn, "test", 2)
        bulk_save_emails(
            conn,
            2,
            [
                Email(
                    sender="x@y.com",
                    recipient="z@w.com",
                    date="2026-01-03",
                    message_id="<3@t>",
                    subject="Hey",
                    body="World",
                ),
            ],
        )
        complete_instance(conn, 2)

        export_path = tmp_path / "output"
        export_corpus(conn, str(export_path), corpus_id="test")

        assert (export_path / "instances" / "instance_0001.jsonl").exists()
        assert (export_path / "instances" / "instance_0002.jsonl").exists()
        assert (export_path / "email_corpus.jsonl").exists()
        assert not (export_path / "instance_1.jsonl").exists()

        combined = (export_path / "email_corpus.jsonl").read_text().strip().split("\n")
        assert len(combined) == 2

