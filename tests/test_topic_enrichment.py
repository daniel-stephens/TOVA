"""Regression checks for replacing model descriptions without retraining."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tova.core.topic_enrichment import enrich_model


class TopicEnrichmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_dir = self.root / "m_test"
        self.folder = self.model_dir / "TMmodel"
        self.folder.mkdir(parents=True)
        self.metadata = self.model_dir / "metadata.json"
        self.metadata.write_text(json.dumps({"corpus_id": "c_test", "name": "Existing model"}))
        corpus = self.root / "c_test"
        corpus.mkdir()
        (corpus / "data.json").write_text(json.dumps([
            {"id": "b", "text": "Second training document"},
            {"id": "a", "text": "First training document"},
        ]))
        (self.folder / "thetas_representation.json").write_text(json.dumps({"a": {}, "b": {}}))
        (self.folder / "tpc_labels.txt").write_text("Topic 0\nTopic 1\n")
        (self.folder / "tpc_summaries.txt").write_text("Old summary 0\nOld summary 1\n")
        (self.model_dir / "model.bin").write_bytes(b"unchanged model weights")
        (self.folder / "thetas.npz").write_bytes(b"unchanged document assignments")
        (self.model_dir / "participant-labels.json").write_text('{"a": "Custom label"}')
        self.originals = {path: path.read_bytes() for path in self.model_dir.rglob("*") if path.is_file()}
        config = patch("tova.core.topic_enrichment.load_yaml_config_file", return_value={
            "general": {"llm_provider": "ollama", "llm_model_type": "gemma3:4b",
                        "llm_server": "http://host.docker.internal:11434"},
        })
        config.start()
        self.addCleanup(config.stop)
        model_patch = patch("tova.core.topic_enrichment.TMmodel")
        self.model_class = model_patch.start()
        self.addCleanup(model_patch.stop)
        self.tm = self.model_class.return_value
        self.tm._thetas = np.ones((2, 2))
        self.tm._ntopics = 2
        self.tm.get_tpc_labels.return_value = [(0, '"Security\nAudit"'), (1, "System Stability")]
        self.tm.get_tpc_summaries.return_value = [(0, "Audit of\nsecurity."), (1, "Stability concerns.")]

    def enrich(self):
        return enrich_model("m_test", drafts_root=self.root)

    def assert_originals_preserved(self):
        for path, content in self.originals.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertFalse((self.model_dir / "enrichment-backups").exists())

    def test_updates_descriptions_with_backups_and_aligned_document_order(self):
        result = self.enrich()
        self.assertEqual(result["labels"], ["Security Audit", "System Stability"])
        self.assertEqual((self.folder / "tpc_labels.txt").read_text(), "Security Audit\nSystem Stability\n")
        self.assertEqual((self.folder / "tpc_summaries.txt").read_text(), "Audit of security.\nStability concerns.\n")
        self.assertEqual(self.model_class.call_args.kwargs["df_corpus_train"]["id"].tolist(), ["a", "b"])
        metadata = json.loads(self.metadata.read_text())
        self.assertEqual(metadata["name"], "Existing model")
        self.assertEqual(metadata["topic_enrichment"]["model"], "gemma3:4b")
        backup = Path(result["backup"])
        for path, content in self.originals.items():
            if path.name in {"metadata.json", "tpc_labels.txt", "tpc_summaries.txt"}:
                self.assertEqual((backup / path.name).read_bytes(), content)
            else:
                self.assertEqual(path.read_bytes(), content)

    def test_summary_failure_preserves_all_previous_artifacts(self):
        self.tm.get_tpc_summaries.side_effect = ConnectionError("Ollama unavailable")
        with self.assertRaisesRegex(ConnectionError, "Ollama unavailable"):
            self.enrich()
        self.assert_originals_preserved()

    def test_incomplete_or_placeholder_outputs_are_not_saved(self):
        cases = [
            ([(0, "Security Audit")], [(0, "Summary."), (1, "Summary.")]),
            ([(0, "Topic 0"), (1, "System Stability")], [(0, "Summary."), (1, "Summary.")]),
            ([(0, "Security Audit"), (1, "System Stability")], [(0, ""), (1, "Summary.")]),
        ]
        for labels, summaries in cases:
            with self.subTest(labels=labels, summaries=summaries):
                self.tm.get_tpc_labels.return_value = labels
                self.tm.get_tpc_summaries.return_value = summaries
                with self.assertRaises(ValueError):
                    self.enrich()
                self.assert_originals_preserved()

    def test_unmatched_document_ids_fail_before_llm_generation(self):
        (self.folder / "thetas_representation.json").write_text('{"missing": {}}')
        with self.assertRaisesRegex(ValueError, "cannot be matched"):
            self.enrich()
        self.model_class.assert_not_called()

    def test_mismatched_matrix_fails_before_llm_generation(self):
        self.tm._thetas = np.ones((3, 2))
        with self.assertRaisesRegex(ValueError, "different lengths"):
            self.enrich()
        self.tm.get_tpc_labels.assert_not_called()
        self.assert_originals_preserved()


if __name__ == "__main__":
    unittest.main()
