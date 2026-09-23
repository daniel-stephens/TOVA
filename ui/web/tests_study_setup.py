"""Local import and provisioning tests; all backend requests are mocked."""

from __future__ import annotations

import io
import json
import stat
import tempfile
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase, override_settings

from web.models import ParticipantSession, StudyGroup, User
from web.services import runtime as R
from web.services.study_setup import load_documents
from web.study_features import get_study_model_id


class DocumentImportTests(SimpleTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / "emails.jsonl"

    def write_source(self, content):
        self.source.write_text(content, encoding="utf-8")
        return self.source

    def test_adjacent_json_objects_and_duplicate_source_ids_preserve_every_record(self):
        records = [
            {"mid": "email-7", "body": "First version."},
            {"mid": "email-7", "body": "Second version."},
            {"mid": "email-8", "body": "Another email."},
        ]
        source = self.write_source(
            json.dumps(records[0]) + json.dumps(records[1]) + "\n" + json.dumps(records[2])
        )
        documents, report = load_documents(source)
        self.assertEqual(len(documents), 3)
        self.assertEqual(len({document["id"] for document in documents}), 3)
        self.assertEqual([document["original_id"] for document in documents], ["email-7", "email-7", "email-8"])
        self.assertEqual([document["text"] for document in documents], [record["body"] for record in records])
        self.assertEqual(report["source_records"], 3)
        self.assertEqual(report["imported_records"], 3)
        self.assertEqual(report["repeated_source_ids"], 1)

    def test_json_array_wrapped_documents_and_jsonl_are_supported(self):
        records = [{"id": "a", "text": "A document."}, {"id": "b", "text": "Another document."}]
        formats = [json.dumps(records), json.dumps({"documents": records}), "\n".join(map(json.dumps, records))]
        for content in formats:
            with self.subTest(content=content):
                documents, report = load_documents(self.write_source(content))
                self.assertEqual([document["text"] for document in documents], ["A document.", "Another document."])
                self.assertEqual([document["original_id"] for document in documents], ["a", "b"])
                self.assertEqual(report["source_records"], 2)

    def test_email_headers_and_literal_newlines_are_readable(self):
        record = {
            "mid": "mail-1", "sender": "alice@example.test", "recipient": "bob@example.test",
            "date": "2026-09-22", "subject": "Trial update",
            "body": "First line.\\nSecond line.\nThird line.",
        }
        documents, _ = load_documents(self.write_source(json.dumps(record)))
        document = documents[0]
        self.assertEqual(document["text"], "Trial update\n\nFirst line.\nSecond line.\nThird line.")
        for header in (
            "Email-ID: mail-1", "From: alice@example.test", "To: bob@example.test",
            "Date: 2026-09-22", "Subject: Trial update",
        ):
            self.assertIn(header, document["raw_text"])
        self.assertTrue(document["raw_text"].endswith("First line.\nSecond line.\nThird line."))
        self.assertEqual(document["sourcefile"], "emails.jsonl")

    def test_normalized_document_retains_original_id_and_raw_text(self):
        record = {"id": "internal-9", "original_id": "email-7", "text": "Training text", "raw_text": "Original email."}
        documents, _ = load_documents(self.write_source(json.dumps([record])))
        self.assertEqual(documents[0]["original_id"], "email-7")
        self.assertEqual(documents[0]["text"], "Training text")
        self.assertEqual(documents[0]["raw_text"], "Original email.")

    def test_numeric_zero_source_ids_are_preserved(self):
        for id_field in ("original_id", "id", "mid"):
            with self.subTest(id_field=id_field):
                documents, _ = load_documents(self.write_source(json.dumps([
                    {id_field: 0, "body": "A document with a zero identifier."},
                ])))
                self.assertEqual(documents[0]["original_id"], "0")

    def test_missing_text_or_body_rejects_the_import(self):
        for record in ({"mid": "a"}, {"mid": "a", "body": "   "}, {"mid": "a", "text": ""}):
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "no text or body"):
                    load_documents(self.write_source(json.dumps(record)))

    def test_limit_must_be_positive_and_reports_full_source_count(self):
        source = self.write_source(json.dumps([
            {"id": str(index), "text": f"Document {index}"} for index in range(3)
        ]))
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "limit must be positive"):
                    load_documents(source, limit=limit)
        documents, report = load_documents(source, limit=2)
        self.assertEqual(len(documents), 2)
        self.assertEqual(report["source_records"], 3)
        self.assertEqual(report["imported_records"], 2)

    def test_empty_invalid_or_non_object_records_are_rejected(self):
        for content in ("", "[]", '{"documents": {}}', "[42]", '{"mid":'):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    load_documents(self.write_source(content))


@override_settings(PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"])
class PrepareStudyCommandTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.drafts = self.root / "drafts"
        self.drafts.mkdir()
        self.original_file = self.drafts / "existing-data.json"
        self.original_file.write_text('{"id": "old", "text": "Keep this data"}\n')
        self.source = self.root / "emails.jsonl"
        self.source.write_text(json.dumps([
            {"mid": "email-1", "subject": "Update", "body": "Safety analysis."},
            {"mid": "email-1", "subject": "Re: Update", "body": "Review completed."},
        ]))
        self.source_content = self.source.read_bytes()
        self.original_content = self.original_file.read_bytes()
        drafts_patch = mock.patch.object(R, "DRAFTS_SAVE", self.drafts)
        drafts_patch.start()
        self.addCleanup(drafts_patch.stop)

        self.existing_group = StudyGroup.objects.create(
            name="Existing cohort", slug="existing-cohort", default_model_id="old-model", enable_active_learning=False,
        )
        self.existing_user = User.objects.create_user(
            email="existing@study.local", password="existing-password", study_group=self.existing_group,
        )
        self.existing_session = ParticipantSession.objects.create(
            user=self.existing_user, study_group=self.existing_group,
            model_id="old-model", stage=ParticipantSession.STAGE_POST,
        )
        self.old_password_hash = self.existing_user.password
        self.stdout = io.StringIO()
        self.documents = None
        self.statuses = [{"status": "running", "message": "Training"}, {"status": "succeeded", "result": {"topics": 3}}]
        self.request_patch = mock.patch(
            "web.management.commands.prepare_study.requests.request", side_effect=self.fake_request,
        )
        self.request = self.request_patch.start()
        self.addCleanup(self.request_patch.stop)
        sleep_patch = mock.patch("web.management.commands.prepare_study.time.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def response(self, body, status=200):
        return mock.Mock(ok=status < 400, status_code=status, text=json.dumps(body), json=mock.Mock(return_value=body))

    def fake_request(self, method, url, **kwargs):
        route = url.removeprefix(R.API)
        # Provisioning must wait until every backend operation succeeds.
        self.assertEqual(StudyGroup.objects.count(), 1)
        self.assertEqual(User.objects.filter(is_staff=False).count(), 1)
        if (method, route) == ("GET", "/health"):
            return self.response({"status": "ok"})
        if (method, route) == ("POST", "/data/datasets"):
            self.documents = kwargs["json"]["documents"]
            return self.response({"draft_id": "new-dataset"})
        if (method, route) == ("GET", "/data/datasets/new-dataset"):
            return self.response({"id": "new-dataset", "documents": self.documents})
        if (method, route) == ("POST", "/data/corpora"):
            return self.response({"draft_id": "new-corpus"})
        if (method, route) == ("GET", "/data/corpora/new-corpus"):
            return self.response({"id": "new-corpus", "documents": self.documents})
        if (method, route) == ("POST", "/train/json"):
            return self.response({"job_id": "new-job", "model_id": "new-model"})
        if (method, route) == ("GET", "/status/jobs/new-job"):
            return self.response(self.statuses.pop(0))
        if (method, route) == ("POST", "/data/corpora/new-corpus/add_model"):
            return self.response({"ok": True})
        if (method, route) == ("GET", "/data/models/new-model"):
            return self.response({"id": "new-model", "corpus_id": "new-corpus"})
        self.fail(f"Unexpected API operation: {method} {route}")

    def run_command(self, **options):
        arguments = {
            "source": str(self.source), "name": "CorpusBio test", "topics": 3,
            "iterations": 20, "task_minutes": 4, "stdout": self.stdout,
        }
        arguments.update(options)
        return call_command("prepare_study", **arguments)

    def run_output(self):
        directories = list((self.root / "study-runs").iterdir())
        self.assertEqual(len(directories), 1)
        output = directories[0]
        report = json.loads((output / "report.json").read_text())
        return output, report

    def assert_originals_unchanged(self):
        self.existing_user.refresh_from_db()
        self.existing_group.refresh_from_db()
        self.existing_session.refresh_from_db()
        self.assertEqual(self.existing_user.password, self.old_password_hash)
        self.assertEqual(self.existing_user.study_group, self.existing_group)
        self.assertEqual(self.existing_group.default_model_id, "old-model")
        self.assertFalse(self.existing_group.enable_active_learning)
        self.assertEqual(self.existing_session.stage, ParticipantSession.STAGE_POST)
        self.assertEqual(self.existing_session.model_id, "old-model")
        self.assertEqual(self.source.read_bytes(), self.source_content)
        self.assertEqual(self.original_file.read_bytes(), self.original_content)

    def test_success_imports_trains_and_provisions_fresh_accounts_with_private_credentials(self):
        self.run_command()
        output, report = self.run_output()
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["imported_records"], 2)
        self.assertEqual(report["repeated_source_ids"], 1)
        self.assertEqual(report["model_id"], "new-model")
        self.assertEqual(report["corpus_id"], "new-corpus")
        self.assertEqual(json.loads((output / "documents.json").read_text()), self.documents)
        credentials_path = output / "credentials.json"
        self.assertEqual(stat.S_IMODE(credentials_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        credentials = json.loads(credentials_path.read_text())
        researcher = User.objects.get(email=credentials["researcher"]["email"])
        participant = User.objects.select_related("study_group").get(email=credentials["participant"]["email"])
        self.assertTrue(researcher.is_superuser)
        self.assertTrue(researcher.is_staff)
        self.assertTrue(researcher.is_admin)
        self.assertTrue(researcher.check_password(credentials["researcher"]["password"]))
        self.assertFalse(participant.is_superuser)
        self.assertFalse(participant.is_staff)
        self.assertFalse(participant.is_admin)
        self.assertTrue(participant.check_password(credentials["participant"]["password"]))
        self.assertEqual(get_study_model_id(participant), "new-model")
        self.assertEqual(participant.study_group.pk, report["group_id"])
        self.assertTrue(participant.study_group.enable_active_learning)
        self.assertEqual(participant.study_group.task_minutes, 4)
        self.assertTrue(participant.participant_code)
        self.assertNotEqual(participant.study_group.pk, self.existing_group.pk)
        self.assertFalse(ParticipantSession.objects.filter(user=participant).exists())
        for account in credentials.values():
            self.assertNotIn(account["password"], self.stdout.getvalue())

        operations = [(call.args[0], call.args[1].removeprefix(R.API)) for call in self.request.call_args_list]
        self.assertEqual(operations, [
            ("GET", "/health"), ("POST", "/data/datasets"), ("GET", "/data/datasets/new-dataset"),
            ("POST", "/data/corpora"), ("GET", "/data/corpora/new-corpus"), ("POST", "/train/json"),
            ("GET", "/status/jobs/new-job"), ("GET", "/status/jobs/new-job"),
            ("POST", "/data/corpora/new-corpus/add_model"), ("GET", "/data/models/new-model"),
        ])
        dataset_payload = self.request.call_args_list[1].kwargs["json"]
        self.assertEqual(dataset_payload["owner_id"], str(researcher.pk))
        training = self.request.call_args_list[5].kwargs["json"]
        self.assertEqual(training["model"], "tomotopyLDA")
        self.assertEqual(training["corpus_id"], "new-corpus")
        self.assertEqual(training["data"], self.documents)
        self.assertEqual(training["training_params"]["num_topics"], 3)
        self.assertEqual(training["training_params"]["num_iters"], 20)
        self.assertIs(training["training_params"]["preprocess_text"], False)
        for setting in ("do_labeller", "do_summarizer"):
            self.assertIs(training["training_params"][setting], True)
        self.assertEqual(self.request.call_args_list[-2].kwargs["params"], {"model_id": "new-model"})
        self.assert_originals_unchanged()

    def test_offline_test_requires_explicit_no_enrich_option(self):
        self.run_command(no_enrich=True)
        training = self.request.call_args_list[5].kwargs["json"]
        for setting in ("do_labeller", "do_summarizer"):
            self.assertIs(training["training_params"][setting], False)

    def test_enrichment_warning_stops_participant_enrollment(self):
        self.statuses = [{"status": "succeeded", "result": {
            "topics": 3, "warnings": ["Topic summaries could not be generated: Ollama unavailable"],
        }}]
        with self.assertRaisesRegex(CommandError, "Ollama unavailable"):
            self.run_command()
        _, report = self.run_output()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(StudyGroup.objects.count(), 1)
        self.assertEqual(User.objects.filter(is_staff=False).count(), 1)
        self.assert_originals_unchanged()

    def test_failed_training_saves_progress_without_enrolling_participant(self):
        self.statuses = [{"status": "failed", "error": "Invalid training input"}]
        with self.assertRaisesRegex(CommandError, "Invalid training input"):
            self.run_command()
        output, report = self.run_output()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["job_id"], "new-job")
        self.assertEqual(report["corpus_id"], "new-corpus")
        credentials = json.loads((output / "credentials.json").read_text())
        self.assertEqual(set(credentials), {"researcher"})
        researcher = User.objects.get(email=credentials["researcher"]["email"])
        self.assertTrue(researcher.check_password(credentials["researcher"]["password"]))
        self.assertEqual(StudyGroup.objects.count(), 1)
        self.assertEqual(User.objects.filter(is_staff=False).count(), 1)
        self.assert_originals_unchanged()

    def test_api_failure_does_not_enroll_participant(self):
        successful_request = self.fake_request

        def failing_request(method, url, **kwargs):
            if url.endswith("/data/corpora"):
                return self.response({"error": "Corpus creation unavailable"}, status=503)
            return successful_request(method, url, **kwargs)

        self.request.side_effect = failing_request
        with self.assertRaisesRegex(CommandError, "HTTP 503"):
            self.run_command()
        _, report = self.run_output()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["dataset_id"], "new-dataset")
        self.assertEqual(StudyGroup.objects.count(), 1)
        self.assertEqual(User.objects.filter(is_staff=False).count(), 1)
        self.assert_originals_unchanged()

    def test_nonpositive_limit_fails_before_any_request_or_provisioning(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(CommandError, "limit must be positive"):
                    self.run_command(limit=limit)
        self.request.assert_not_called()
        self.assertEqual(User.objects.count(), 1)
        self.assertEqual(StudyGroup.objects.count(), 1)
        self.assertFalse((self.root / "study-runs").exists())
        self.assert_originals_unchanged()

    def test_nonpositive_training_configuration_fails_before_provisioning(self):
        for option in ("topics", "iterations", "task_minutes", "timeout"):
            with self.subTest(option=option):
                with self.assertRaisesRegex(CommandError, "must be positive"):
                    self.run_command(**{option: 0})
        self.request.assert_not_called()
        self.assertEqual(User.objects.count(), 1)
        self.assertFalse((self.root / "study-runs").exists())
