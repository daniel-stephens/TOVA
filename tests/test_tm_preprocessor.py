import unittest
from pathlib import Path
import pandas as pd
from tova.preprocessing.tm_preprocessor import TMPreprocessor


class TestTMPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TMPreprocessor(max_df=1.0, min_df=1)

        self.sample_data = pd.DataFrame({
            "id": [1, 2, 3],
            "text": [
                "This is a test document.",
                "Another document for testing purposes.",
                "Final test document with more content."
            ]
        })

    def test_lemmatization(self):
        lemmas = self.preprocessor._lemmatize("This is a testing example.")
        self.assertIsInstance(lemmas, list)
        self.assertGreater(len(lemmas), 0)

    def test_fit_transform(self):
        result = self.preprocessor.fit_transform(
            self.sample_data, text_col="text", id_col="id")
        self.assertIn("id", result.columns)
        self.assertIn("text", result.columns)
        self.assertIn("lemmas", result.columns)
        self.assertIn("bow", result.columns)
        self.assertIn("tfidf", result.columns)
        self.assertEqual(len(result), len(self.sample_data))

    def test_transform_new(self):
        self.preprocessor.fit_transform(
            self.sample_data, text_col="text", id_col="id")

        new_texts = ["A completely new document.", "Another example text."]
        transformed = self.preprocessor.transform_new(new_texts)
        self.assertEqual(transformed.shape[0], len(new_texts))

    def test_stopwords_loading(self):
        stopwords_file = Path("test_stopwords.txt")
        stopwords_file.write_text("stopword1\nstopword2\n")
        stopwords = self.preprocessor._load_stopwords([stopwords_file])
        self.assertIn("stopword1", stopwords)
        self.assertIn("stopword2", stopwords)
        stopwords_file.unlink()

    def test_equivalents_loading(self):
        equivalents_file = Path("tests/test_equivalents.txt")
        equivalents_file.write_text("word1:replacement1\nword2:replacement2\n")
        equivalents = self.preprocessor._load_equivalents([equivalents_file])
        self.assertEqual(equivalents.get("word1"), "replacement1")
        self.assertEqual(equivalents.get("word2"), "replacement2")
        equivalents_file.unlink()

    def test_preprocess_bills_sample(self):
        file_path = "data_test/bills_sample_100.csv"
        df = pd.read_csv(file_path)

        result = self.preprocessor.fit_transform(
            df, text_col="summary", id_col="id")

        self.assertIn("id", result.columns)
        self.assertIn("text", result.columns)
        self.assertIn("lemmas", result.columns)
        self.assertIn("bow", result.columns)
        self.assertIn("tfidf", result.columns)
        self.assertEqual(len(result), len(df))

        print(result.columns)

    def test_embeddings(self):
        preprocessor_with_embeddings = TMPreprocessor(
            max_df=1.0, min_df=1, do_embeddings=True, embeddings_model="all-MiniLM-L6-v2"
        )

        sample_data = pd.DataFrame({
            "id": [1, 2],
            "text": [
                "This is a test document.",
                "Another document for testing embeddings."
            ]
        })

        result = preprocessor_with_embeddings.fit_transform(
            sample_data, text_col="text", id_col="id", compute_embeddings=True
        )

        self.assertIn("embedding", result.columns)
        self.assertEqual(len(result), len(sample_data))
        self.assertTrue(all(result["embedding"].apply(lambda x: len(x) > 0)))

if __name__ == "__main__":
    unittest.main()
