"""Local LDA regression checks; synthetic documents, no external model calls."""

import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from tova.core import drafts, models
from tova.core.dispatchers import get_model_info_dispatch, get_thetas_documents_by_id_dispatch
from tova.preprocessing.tm_preprocessor import TMPreprocessor
from tova.topic_models.models.traditional.tomotopy_lda_tm_model import TomotopyLDATMmodel
from tova.topic_models.tm_model import TMmodel


class TestLocalLDAPipeline(unittest.TestCase):
    def test_embeddings_load_only_when_requested_and_receive_full_text(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump({"tm_preprocessor": {
                "spacy": {"spacy_model": "test-spacy"},
                "embeddings": {"model_name": "test-encoder"},
            }}))
            with patch("tova.preprocessing.tm_preprocessor.load_spacy") as spacy_loader, \
                 patch("tova.preprocessing.tm_preprocessor.SentenceTransformer") as encoder:
                spacy_loader.return_value.Defaults.stop_words = set()
                preprocessor = TMPreprocessor(config_path=config_path,
                                              logger=logging.getLogger(__name__))
                encoder.assert_not_called()
                texts = ["garden flower", "movie actor"]
                encoder.return_value.encode.return_value = np.ones((2, 3))
                with patch.object(preprocessor, "_lemmatize", return_value=["token"]):
                    result = preprocessor.fit_transform(
                        pd.DataFrame({"id": ["a", "b"], "text": texts}),
                        compute_bow=False, compute_tfidf=False, compute_embeddings=True,
                    )
                encoder.assert_called_once_with("test-encoder")
                self.assertEqual(encoder.return_value.encode.call_args.args[0], texts)
                self.assertEqual(len(result["embeddings"]), 2)

    def test_preprocessed_corpus_trains_reloads_and_provides_query_outputs(self):
        documents = [
            {"id": "doc_1", "text": "movie actor cinema film drama screen"},
            {"id": "doc_2", "text": "actor film cinema movie screen drama"},
            {"id": "doc_3", "text": "movie cinema actor screen film drama"},
            {"id": "doc_4", "text": "garden plant flower tree green soil"},
            {"id": "doc_5", "text": "plant flower soil garden tree green"},
            {"id": "doc_6", "text": "tree plant flower green garden soil"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "logger": {"dir_logger": str(root / "logs"), "file_log": False},
                "topic_modeling": {
                    "traditional": {"num_topics": 2, "do_labeller": False,
                                    "do_summarizer": False},
                    "tomotopy": {"num_iters": 10, "iter_interval": 5},
                },
            }))
            model_dir = root / "m_local"
            # A disabled preprocessor must never load spaCy or embeddings.
            with patch("tova.topic_models.models.traditional.base.TMPreprocessor",
                       side_effect=AssertionError("Unexpected preprocessing")), \
                 patch("tova.topic_models.tm_model.Prompter",
                       side_effect=AssertionError("Unexpected LLM call")):
                model = TomotopyLDATMmodel(
                    model_name="Local movie baseline", corpus_id="c_local", id="m_local",
                    model_path=model_dir, config_path=config_path,
                    logger=logging.getLogger(__name__), preprocess_text=False,
                )
                duration, _ = model.train_model(documents)
                self.assertGreater(duration, 0)
                self.assertEqual(sparse.load_npz(model_dir / "TMmodel/thetas.npz").shape,
                                 (len(documents), 2))
                metadata = json.loads((model_dir / "metadata.json").read_text())
                self.assertEqual(metadata["name"], "Local movie baseline")
                self.assertFalse(metadata["tr_params"]["preprocess_text"])

                restored = TomotopyLDATMmodel.from_saved_model(str(model_dir))
                self.assertEqual(restored.model_path, model_dir)
                self.assertEqual(restored.model_name, model.model_name)
                self.assertFalse(restored.preprocess_text)
                thetas, _ = restored.infer(documents[:2])
                self.assertEqual(thetas.shape, (2, 2))
                np.testing.assert_allclose(thetas.sum(axis=1), 1, atol=1e-6)

            model_info = get_model_info_dispatch(str(model_dir), config_path=config_path)
            self.assertEqual(len(model_info["Topics Info"]), 2)
            # FastAPI's JSON encoder rejects NaN and Infinity. A small model
            # must not return its own topic with -Infinity similarity.
            json.dumps(model_info, allow_nan=False)
            for topic_id, topic in model_info["Topics Info"].items():
                self.assertEqual(len(topic["Similar Topics (Coocurring)"]), 1)
                self.assertNotEqual(topic_id, topic["Similar Topics (Coocurring)"][0]["ID"])
            theta_info = get_thetas_documents_by_id_dispatch(
                ["doc_1", "doc_6"], str(model_dir), config_path=config_path)
            self.assertEqual(set(theta_info), {"doc_1", "doc_6"})

            with patch.object(drafts, "DRAFTS_SAVE", root):
                self.assertEqual(models.list_models("c_local")[0].name, model.model_name)
                self.assertEqual(models.list_models("c_other"), [])

    def test_constant_topic_weights_have_finite_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            model = TMmodel(Path(directory), do_labeller=False)
            model._thetas = sparse.csr_matrix([[0.5, 0.5], [0.5, 0.5]])
            model._betas = np.array([[0.6, 0.4], [0.4, 0.6]])
            model._ntopics = 2
            similarities = model.getSimilarTopicsDicts(nsimilar=5)
            json.dumps(similarities, allow_nan=False)
            self.assertEqual(similarities["Coocurring"][0], [(1, 0.0)])


if __name__ == "__main__":
    unittest.main()
