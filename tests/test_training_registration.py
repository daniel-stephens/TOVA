"""Successful API jobs publish corpus ownership and association together."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks

from tova.api.jobs.domain import JobStatus
from tova.api.jobs.store import InMemoryJobStore
from tova.api.models.data_schemas import Corpus, DataRecord, Dataset
from tova.api.routers import data_handling, training
from tova.core import corpora, drafts, models


class TestTrainingRegistration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        corpus_dir = self.root / "c_test"
        corpus_dir.mkdir()
        (corpus_dir / "metadata.json").write_text(json.dumps({
            "name": "Test corpus", "owner_id": "participant-one", "models": [],
        }))
        (corpus_dir / "data.json").write_text("[]")
        for target, value in [
            ("tova.core.drafts.DRAFTS_SAVE", self.root),
            ("tova.api.routers.training.DRAFTS_SAVE", self.root),
            ("tova.api.routers.training.job_store", InMemoryJobStore()),
            ("tova.api.routers.training.cancellation_tokens", {}),
        ]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def save_fake_training_outputs(self, **kwargs):
        directory = Path(kwargs["output"])
        directory.mkdir()
        (directory / "metadata.json").write_text(json.dumps({
            "name": kwargs["model_name"], "corpus_id": kwargs["corpus_id"],
            "created_at": "2026-09-23T00:00:00Z", "tr_params": {},
        }))
        return 0.1, []

    async def enqueue(self, background, corpus_id="c_test"):
        return await training._enqueue_training_job(
            model="tomotopyLDA", data=[{"id": "doc_test", "text": "local corpus text"}],
            model_name="Local model", corpus_id=corpus_id,
            config_path="static/config/config.yaml", training_params={}, bg=background,
        )

    async def test_success_publishes_owner_and_corpus_link_before_job_success(self):
        background = BackgroundTasks()
        response = await self.enqueue(background)
        with patch.object(training, "train_model_dispatch", self.save_fake_training_outputs):
            await background()

        job = await training.job_store.get(response.job_id)
        self.assertEqual(job.status, JobStatus.succeeded)
        self.assertEqual(job.owner_id, "participant-one")
        saved_model = models.get_model(response.model_id)
        self.assertEqual(saved_model.owner_id, "participant-one")
        self.assertEqual(saved_model.metadata["owner_id"], "participant-one")
        self.assertEqual([model.id for model in corpora.list_corpus_models("c_test")],
                         [response.model_id])

        # The existing UI follows job completion with this request.
        for _ in range(2):
            result = data_handling.add_model_to_corpus("c_test", response.model_id)
            self.assertEqual(result.status_code, 201)
        self.assertEqual(corpora.get_corpus("c_test").models, [response.model_id])

    async def test_failed_training_does_not_associate_a_model(self):
        background = BackgroundTasks()
        response = await self.enqueue(background)
        with patch.object(training, "train_model_dispatch", side_effect=ValueError("training failed")):
            await background()
        self.assertEqual((await training.job_store.get(response.job_id)).status, JobStatus.failed)
        self.assertEqual(corpora.get_corpus("c_test").models, [])

    async def test_unknown_corpus_is_rejected_before_a_job_is_created(self):
        with self.assertRaisesRegex(ValueError, "Training corpus not found"):
            await self.enqueue(BackgroundTasks(), corpus_id="c_missing")
        self.assertEqual(await training.job_store.list(), [])

    def test_corpus_preserves_source_ids_and_allocates_distinct_document_ids(self):
        imported = Corpus(name="Imported emails", owner_id="participant-one", datasets=[
            Dataset(documents=[
                DataRecord(id="source_000001", original_id="email-42", text="First email"),
                DataRecord(id="source_000002", original_id="email-42", text="Duplicate source ID"),
                DataRecord(id="legacy-id", text="Legacy record"),
            ]),
        ])
        response = corpora.create_corpus(imported)
        saved = corpora.get_corpus(response.draft_id)
        self.assertEqual([doc.original_id for doc in saved.documents],
                         ["email-42", "email-42", "legacy-id"])
        self.assertEqual(len({doc.id for doc in saved.documents}), 3)


if __name__ == "__main__":
    unittest.main()
