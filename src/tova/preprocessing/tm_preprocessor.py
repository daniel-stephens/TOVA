import json
import logging
import pathlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy_download import load_spacy # type: ignore
from tova.utils.common import load_yaml_config_file, init_logger  # type: ignore


class TMPreprocessor:
    """
    Text preprocessor that returns:
      - 'text'       : original text
      - 'lemmas'     : spaCy lemmatization + POS filter + stopwords
      - 'bow'        : scipy.sparse.csr_matrix row per doc (stored per-row)
      - 'tfidf'      : scipy.sparse.csr_matrix row per doc (stored per-row)
      - 'embedding'  : (optional) vector per doc using SentenceTransformers
    """

    def __init__(
        self,
        *,
        stopword_files: Optional[Sequence[Union[str, Path]]] = None,
        equivalents_files: Optional[Sequence[Union[str, Path]]] = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("./static/config/config.yaml"),
        **kwargs
    ):  
        
        self._logger = logger if logger else init_logger(config_path, __name__)
        self._logger.info("Initializing TMPreprocessor with provided configurations.")

        config = load_yaml_config_file(config_path, "tm_preprocessor", logger)
        config = {**config, **kwargs}

        spacy_model = config.get("spacy", {}).get("spacy_model")
        spacy_disable = config.get("spacy", {}).get("spacy_disable")
        valid_pos = config.get("spacy", {}).get("valid_pos")
        min_df = config.get("vectorization", {}).get("min_df")
        max_df = config.get("vectorization", {}).get("max_df")
        max_features = config.get("vectorization", {}).get("max_features")
        embeddings_model = config.get("embeddings", {}).get("model_name")
        
        self._logger.debug(f"spaCy model: {spacy_model}, Disabled components: {spacy_disable}")
        self._logger.debug(f"Valid POS tags: {valid_pos}")
        self._logger.debug(f"Stopword files: {stopword_files}")
        self._logger.debug(f"Equivalents files: {equivalents_files}")
        self._logger.debug(f"Min DF: {min_df}, Max DF: {max_df}, Max features: {max_features}")
        self._logger.debug(f"Embeddings model: {embeddings_model}")

        self._logger.info("Loaded parameters from configuration file.")

        try:
            self.nlp = load_spacy(spacy_model, disable=list(spacy_disable))
        except OSError as e:
            raise OSError(
                f"spaCy model '{spacy_model}' is not installed. "
                f"Install it with: python -m spacy download {spacy_model}"
            ) from e

        self.valid_pos = set(valid_pos)

        # stopwords
        self.stopwords = {w.lower() for w in self.nlp.Defaults.stop_words}
        self.stopwords |= self._load_stopwords(stopword_files or [])

        # equivalents
        self.equivalents = self._load_equivalents(equivalents_files or {})

        # vectorizers
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self._cv: Optional[CountVectorizer] = None
        self._tfidf: Optional[TfidfTransformer] = None

        # embeddings
        self._st = None
        if embeddings_model:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers not installed, but embeddings_model was provided.")
            self._st = SentenceTransformer(embeddings_model)

        self._logger.info("TMPreprocessor initialized successfully.")

    @staticmethod
    def _load_stopwords(files: Sequence[Union[str, Path]]) -> set[str]:
        out: set[str] = set()
        for f in files:
            p = Path(f)
            if not p.exists():
                continue
            if p.suffix.lower() == ".json":
                with p.open("r", encoding="utf8") as fin:
                    data = json.load(fin)
                    out |= {w.strip().lower()
                            for w in data.get("wordlist", []) if w.strip()}
            else:
                with p.open("r", encoding="utf8") as fin:
                    out |= {line.strip().lower()
                            for line in fin if line.strip()}
        return out

    @staticmethod
    def _load_equivalents(files: Sequence[Union[str, Path]]) -> Dict[str, str]:
        eq: Dict[str, str] = {}

        def parse(line: str) -> Optional[Tuple[str, str]]:
            line = line.strip()
            if ":" not in line:
                return None
            a, b = line.split(":", 1)
            a, b = a.strip(), b.strip()
            return (a, b) if a and b else None

        for f in files:
            p = Path(f)
            if not p.exists():
                continue
            if p.suffix.lower() == ".json":
                with p.open("r", encoding="utf8") as fin:
                    data = json.load(fin).get("wordlist", [])
                    pairs = filter(None, (parse(x) for x in data))
            else:
                with p.open("r", encoding="utf8") as fin:
                    pairs = list(filter(None, (parse(x) for x in fin)))
            for a, b in pairs:
                eq[a] = b
        return eq

    def _lemmatize(self, text: str) -> List[str]:
        self._logger.debug(f"Lemmatizing text: {text}")
        """
        - lowercase input
        - whitespace split for early mapping
        - apply equivalents
        - spaCy: keep alpha tokens with POS in valid_pos, remove stopwords
        - return lemmas (lowercased)
        """
        toks = text.lower().split()
        toks = [self.equivalents.get(t, t) for t in toks]

        doc = self.nlp(" ".join(toks))
        keep = []
        sw = self.stopwords
        vp = self.valid_pos
        for t in doc:
            lemma = t.lemma_.lower()
            if t.is_alpha and t.pos_ in vp and (not t.is_stop) and lemma not in sw:
                keep.append(lemma)
        self._logger.debug(f"Lemmatized tokens: {keep}")
        return keep

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_col: str = "raw_text",
        id_col: str = "id",
        *,
        compute_bow: bool = True,
        compute_tfidf: bool = True,
        compute_embeddings: bool = False,
    ) -> pd.DataFrame:
        self._logger.info("Starting fit_transform process.")
        self._logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        self._logger.debug(f"Text column: {text_col}, ID column: {id_col}")
        self._logger.debug(f"Compute BoW: {compute_bow}, Compute TF-IDF: {compute_tfidf}, Compute Embeddings: {compute_embeddings}")

        """Returns a DataFrame with columns: id, text, lemmas, bow, tfidf, (embedding)."""
        out = pd.DataFrame()
        if id_col in df.columns:
            out["id"] = df[id_col].values
        else:
            out["id"] = range(len(df))
        out["raw_text"] = df[text_col].fillna("").astype(str)

        out["lemmas"] = [self._lemmatize(t) for t in out["raw_text"].tolist()]

        # BoW
        if compute_bow:
            self._cv = CountVectorizer(
                tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False,
                min_df=self.min_df, max_df=self.max_df, max_features=self.max_features,
                token_pattern=None
            )
            X_bow = self._cv.fit_transform(out["lemmas"].tolist())
            out["bow"] = [X_bow.getrow(i) for i in range(X_bow.shape[0])]

        # TF-IDF
        if compute_tfidf:
            if self._cv is None:
                raise RuntimeError(
                    "fit_transform() must compute BoW before TF-IDF.")
            X_bow = sparse.vstack(out["bow"].to_list())
            self._tfidf = TfidfTransformer(norm="l2", use_idf=True).fit(X_bow)
            X_tfidf = self._tfidf.transform(X_bow)
            out["tfidf"] = [X_tfidf.getrow(i) for i in range(X_tfidf.shape[0])]

        # Embeddings
        if compute_embeddings:
            if self._st is None:
                raise RuntimeError("Embeddings model not initialized.")
            texts = [" ".join(toks) for toks in out["lemmas"]]
            embs = self._st.encode(
                texts, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True
            )
            out["embeddings"] = list(embs)

        self._logger.info("fit_transform process completed successfully.")
        return out

    def transform_new(self, texts: Iterable[str], *, return_tfidf: bool = True) -> sparse.csr_matrix:
        self._logger.info("Starting transform_new process.")
        self._logger.debug(f"Number of texts to transform: {len(texts)}")
        if not self._cv:
            raise RuntimeError(
                "Vectorizer not fitted. Call fit_transform() first.")
        lemmas = [self._lemmatize(t if isinstance(
            t, str) else str(t)) for t in texts]
        X_bow = self._cv.transform(lemmas)
        if return_tfidf:
            if not self._tfidf:
                self._tfidf = TfidfTransformer(
                    norm="l2", use_idf=True).fit(X_bow)
            self._logger.info("transform_new process completed successfully.")
            return self._tfidf.transform(X_bow)
        self._logger.info("transform_new process completed successfully.")
        return X_bow
