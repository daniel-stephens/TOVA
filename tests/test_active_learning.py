"""Participant isolation through the AL API and persisted session state.

Run without loading the model-training services:
    PYTHONPATH=src python -m unittest discover -s tests -p test_active_learning.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tova.api.routers.active_learning import router
from tova.core import active_learning as al


class ActiveLearningIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.drafts = Path(self.temp_dir.name)
        corpus_dir = self.drafts / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "data.json").write_text(json.dumps([
            {"id": "1", "text": "Railway routes and train stations"},
            {"id": "2", "text": "Hospital doctors and medical services"},
            {"id": "3", "text": "Railway ticket booking service"},
        ]), encoding="utf-8")
        self.model_dir = self.drafts / "model"
        self.model_dir.mkdir()
        (self.model_dir / "metadata.json").write_text(
            json.dumps({"corpus_id": "corpus", "type": "traditional"}),
            encoding="utf-8",
        )
        drafts_patch = patch.object(al, "DRAFTS_SAVE", self.drafts)
        drafts_patch.start()
        self.addCleanup(drafts_patch.stop)
        al._sessions.clear()
        self.addCleanup(al._sessions.clear)

        app = FastAPI()
        app.include_router(router, prefix="/active-learning")
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.base_url = "/active-learning/model"

    def label(self, doc_id, label, user_id=None):
        payload = {"doc_id": doc_id, "label": label}
        if user_id is not None:
            payload["user_id"] = user_id
        response = self.client.post(f"{self.base_url}/label", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_label_recommend_status_and_undo_are_participant_scoped(self):
        self.label("1", "Transport", "alice")
        self.label("2", "Health", "alice")

        alice_status = self.client.get(
            f"{self.base_url}/status", params={"user_id": "alice"}
        ).json()
        bob_status = self.client.get(
            f"{self.base_url}/status", params={"user_id": "bob"}
        ).json()
        self.assertEqual(alice_status["num_labeled"], 2)
        self.assertTrue(alice_status["classifier_trained"])
        self.assertEqual(bob_status["num_labeled"], 0)
        self.assertFalse(bob_status["classifier_trained"])

        alice_rec = self.client.get(
            f"{self.base_url}/recommend", params={"user_id": "alice"}
        ).json()
        bob_rec = self.client.get(
            f"{self.base_url}/recommend", params={"user_id": "bob"}
        ).json()
        self.assertEqual(alice_rec["doc_id"], "3")
        self.assertEqual(alice_rec["current_labels"], {"1": "Transport", "2": "Health"})
        self.assertTrue(alice_rec["classifier_trained"])
        self.assertEqual(bob_rec["current_labels"], {})
        self.assertFalse(bob_rec["classifier_trained"])

        # The same document can have independently chosen human labels.
        self.label("1", "Personal", "bob")
        removed = self.client.delete(
            f"{self.base_url}/label/1", params={"user_id": "alice"}
        )
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json()["current_labels"], {"2": "Health"})
        self.assertFalse(removed.json()["classifier_trained"])
        bob_labels = self.client.get(
            f"{self.base_url}/labels", params={"user_id": "bob"}
        ).json()
        self.assertEqual(bob_labels["labels"], {"1": "Personal"})

    def test_scoped_labels_survive_cache_clear_without_leaking_to_legacy(self):
        self.label("1", "Transport", "alice")
        self.label("2", "Health", "bob")
        al.clear_session("model")
        self.assertFalse(al._sessions)

        for user_id, expected in (("alice", {"1": "Transport"}), ("bob", {"2": "Health"})):
            response = self.client.get(
                f"{self.base_url}/labels", params={"user_id": user_id}
            )
            self.assertEqual(response.json()["labels"], expected)
        self.assertEqual(self.client.get(f"{self.base_url}/labels").json()["labels"], {})

    def test_legacy_labels_keep_their_location_and_are_not_inherited(self):
        legacy_dir = self.model_dir / "active_learning"
        legacy_dir.mkdir()
        (legacy_dir / "labels.json").write_text(
            json.dumps({"1": "Existing legacy label"}), encoding="utf-8"
        )
        self.assertEqual(
            self.client.get(f"{self.base_url}/labels").json()["labels"],
            {"1": "Existing legacy label"},
        )
        self.assertEqual(
            self.client.get(f"{self.base_url}/labels", params={"user_id": "alice"}).json()["labels"],
            {},
        )
        self.label("2", "Existing legacy label")
        self.assertEqual(
            json.loads((legacy_dir / "labels.json").read_text(encoding="utf-8")),
            {"1": "Existing legacy label", "2": "Existing legacy label"},
        )

    def test_participant_identity_is_path_safe_and_does_not_collapse(self):
        identities = ("../../outside", "/absolute/path", "alice/bob", "alice_bob", "用户")
        sessions = [al.get_session("model", user_id=value) for value in identities]
        namespace_root = self.model_dir / "active_learning" / "participants"
        self.assertEqual(len({session._al_dir for session in sessions}), len(identities))
        for session in sessions:
            self.assertEqual(session._al_dir.parent, namespace_root)
            session.add_label("1", "Transport")
            self.assertTrue((session._al_dir / "labels.json").is_file())
        self.assertFalse((self.drafts / "outside").exists())

    def test_clear_one_participant_keeps_other_cached_sessions(self):
        alice = al.get_session("model", user_id="alice")
        bob = al.get_session("model", user_id="bob")
        legacy = al.get_session("model")
        al.clear_session("model", user_id="alice")
        self.assertIsNot(al.get_session("model", user_id="alice"), alice)
        self.assertIs(al.get_session("model", user_id="bob"), bob)
        self.assertIs(al.get_session("model"), legacy)

    def test_completing_one_participant_does_not_complete_another(self):
        for doc_id in ("1", "2", "3"):
            result = self.label(doc_id, "Transport", "alice")
        self.assertEqual(result["doc_id"], "")
        self.assertEqual(
            self.client.get(f"{self.base_url}/recommend", params={"user_id": "alice"}).status_code,
            409,
        )
        self.assertEqual(
            self.client.get(f"{self.base_url}/recommend", params={"user_id": "bob"}).status_code,
            200,
        )
        self.assertEqual(self.client.get(f"{self.base_url}/recommend").status_code, 200)

    def test_revising_labels_to_one_class_clears_the_old_classifier(self):
        self.label("1", "Transport", "alice")
        result = self.label("2", "Health", "alice")
        self.assertTrue(result["classifier_trained"])
        result = self.label("2", "Transport", "alice")
        self.assertFalse(result["classifier_trained"])
        status = self.client.get(
            f"{self.base_url}/status", params={"user_id": "alice"}
        ).json()
        self.assertFalse(status["classifier_trained"])
        self.assertEqual(status["class_distribution"], {"Transport": 2})


if __name__ == "__main__":
    unittest.main()
