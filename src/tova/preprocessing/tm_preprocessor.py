import logging
import pathlib
import spacy
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # type: ignore
from scipy import sparse
from typing import Iterable, List, Optional, Sequence, Tuple, Union
# from spacy_download import load_spacy
from tova.utils.common import load_yaml_config_file


class TMPreprocessor(object):
    """Topic Modeling Preprocessor

    This class handles:
    - NLP preprocessing (tokenization, lemmatization, stopword removal)
    - Document vectorization (BoW, TF-IDF)
    - Contextualized embeddings calculation
    """

    def __init__(
        self,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path(
            "./static/config/config.yaml"),
    ) -> None:
        """Initialize the TMPreprocessor.

        Parameters
        ----------.
        logger : logging.Logger, optional
            Logger object to log activity.
        config_path : pathlib.Path, optional
            Path to the configuration file.
        """

        self._logger = logger if logger else init_logger(config_path, __name__)

        self.config = load_yaml_config_file(
            config_path, "topic_modeling", logger)
        nlp_cfg = self.config.get("nlp", {})

        spacy_model = nlp_cfg.get("spacy_model", "en_core_web_sm")
        disable = nlp_cfg.get("spacy_disable", ["ner", "parser"])
        try:
            self._nlp = spacy.load(spacy_model, disable=disable)
        except OSError:
            # try download hint
            raise OSError(
                f"spaCy model '{spacy_model}' is not installed. "
                f"Install it with: python -m spacy download {spacy_model}"
            )

        # POS + stopwords
        self._valid_pos = set(nlp_cfg.get(
            "valid_pos", ["VERB", "NOUN", "ADJ", "PROPN"]))
        self._stw_list = set()
        # self._prepare_stopwords(nlp_cfg.get("extra_stopwords_files") or nlp_cfg.get("extra_stopwords") or [])

        # Placeholders for cached results
        self._processed_docs: Optional[List[List[str]]] = None
        self._bow_vectorizer: Optional[CountVectorizer] = None
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._bow_matrix: Optional[sparse.spmatrix] = None
        self._tfidf_matrix: Optional[sparse.spmatrix] = None

        # SentenceTransformer model placeholder (lazy)
        self._st_model: Optional[SentenceTransformer] = None

    def _prepare_stopwords(self, stw_files_or_list: Union[Sequence[str], Sequence[pathlib.Path]]) -> None:
        """Load extra stopwords (one token per line) from files or accept a list of strings."""
        # Add spaCy's built-in stopwords first
        try:
            self._stw_list.update(self._nlp.Defaults.stop_words)
        except Exception:
            pass

        if not stw_files_or_list:
            return

        # If user passed a list of tokens directly (not files)
        if any(isinstance(x, str) and "\n" not in x and not pathlib.Path(x).exists() for x in stw_files_or_list):
            # treat as tokens
            for tok in stw_files_or_list:
                if isinstance(tok, str) and "\n" not in tok and not pathlib.Path(tok).exists():
                    self._stw_list.add(tok.strip().lower())
            # also open valid paths if any
        for entry in stw_files_or_list:
            try:
                p = pathlib.Path(entry)
                if p.exists() and p.is_file():
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            w = line.strip().lower()
                            if w:
                                self._stw_list.add(w)
            except Exception as e:
                self._logger.warning(
                    f"Could not load stopwords from {entry}: {e}")

    def preprocess_one(self, rawtext) -> List[str]:
        """
        Implements the preprocessing pipeline, by carrying out:
        - Lemmatization according to POS
        - Removal of non-alphanumerical tokens
        - Removal of basic English stopwords and additional ones provided within stw_files
        - Word tokenization
        - Lowercase conversion

        Parameters
        ----------
        rawtext: str
            Text to preprocess

        Returns
        -------
        final_tokenized: List[str]
            List of tokens (strings) with the preprocessed text
        """
        valid_POS = self._valid_pos

        doc = self._nlp(rawtext)
        lemmatized = [
            token.lemma_
            for token in doc
            if token.is_alpha
            and token.pos_ in valid_POS
            and not token.is_stop
            and token.lemma_.lower() not in self._stw_list
        ]

        # Convert to lowercase
        final_tokenized = [token.lower() for token in lemmatized]

        return final_tokenized

    def preprocess(
        self,
        docs: Iterable[str],
        *,
        keep_empty: bool = False,
        cache: bool = True,
    ) -> List[List[str]]:
        """
        Preprocess a collection of documents.

        Parameters
        ----------
        docs : Iterable[str]
            Raw documents.
        keep_empty : bool, default False
            If False, empty token lists are replaced by [''] so that downstream vectorizers don't drop rows.
        cache : bool, default True
            If True, stores results in self._processed_docs.

        Returns
        -------
        List[List[str]]
            Tokenized and cleaned documents.
        """
        processed: List[List[str]] = []
        for d in docs:
            toks = self.preprocess_one(d if isinstance(d, str) else str(d))
            if not toks and not keep_empty:
                toks = [""]  # preserve row alignment for vectorizers
            processed.append(toks)

        if cache:
            self._processed_docs = processed
        return processed

    def _identity_analyzer(self, doc_tokens: List[str]) -> List[str]:
        """Analyzer for scikit-learn that passes through already tokenized docs."""
        return doc_tokens

    def _ensure_st_model(
        self,
        model_name: Optional[str] = None,
    ) -> SentenceTransformer:
        emb_cfg = self._cfg.get("embeddings", {})
        chosen = model_name or emb_cfg.get("model_name", "all-MiniLM-L6-v2")
        if self._st_model is None or getattr(self._st_model, "model_card", None) != chosen:
            # SentenceTransformer doesn't expose 'model_card' like this; we just reload if name differs
            # Track chosen name to avoid unnecessary reloads
            self._st_model = SentenceTransformer(chosen)
            self._st_model._tm_chosen_name = chosen  # stash
        elif getattr(self._st_model, "_tm_chosen_name", None) != chosen:
            self._st_model = SentenceTransformer(chosen)
            self._st_model._tm_chosen_name = chosen
        return self._st_model

    def get_contextualized_embeddings(
        self,
        docs: Optional[Iterable[str]] = None,
        *,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        return_numpy: bool = True,
        use_preprocessed_text: bool = False,
    ):
        """
        @ TODO: Add docc
        """
        emb_cfg = self._cfg.get("embeddings", {})
        batch_size = batch_size if batch_size is not None else int(
            emb_cfg.get("batch_size", 32))
        normalize = normalize if normalize is not None else bool(
            emb_cfg.get("normalize", True))
        model = self._ensure_st_model(model_name)

        # Determine input texts
        if docs is None:
            if use_preprocessed_text:
                if self._processed_docs is None:
                    raise ValueError(
                        "No cached preprocessed docs. Pass `docs` or call preprocess() first.")
                texts = [
                    " ".join(toks) if toks else "" for toks in self._processed_docs]
            else:
                raise ValueError(
                    "When docs=None, set use_preprocessed_text=True to encode cached tokens.")
        else:
            if use_preprocessed_text:
                tokenized_docs = self.preprocess(docs, cache=False)
                texts = [
                    " ".join(toks) if toks else "" for toks in tokenized_docs]
            else:
                texts = [d if isinstance(d, str) else str(d) for d in docs]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=return_numpy,
        )
        return embeddings
