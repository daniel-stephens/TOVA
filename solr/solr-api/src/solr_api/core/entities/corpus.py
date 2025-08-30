"""
This module implements a class to manage and hold all the information associated with a logical corpus.

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modified: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Project))
Modified: 15/08/2025 (Updated for TOVA Project)
"""

from typing import Iterable, List, Union
import pathlib
import logging
import math

import pandas as pd  # type: ignore
from tova.utils.tm_utils import read_dataframe


def _json_safe(value):
    """
    Convert values that are not JSON-compliant into safe equivalents:
    - NaN/NaT -> None
    - +/- Infinity -> None
    Leaves everything else unchanged.
    """
    # pandas/numpy missing values
    if pd.isna(value):
        return None
    # guard infinities (also covers numpy floats)
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return None
    return value


class Corpus(object):
    """
    A class to manage and hold all the information associated with a logical corpus.
    """

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        corpus_name: str,
        metadata: dict | None = None,
        logger: logging.Logger | None = None,
        # config_file: str = "/config/config.cf"
    ) -> None:
        """Init method.

        Parameters
        ----------
        data : str | pandas.DataFrame
            Path to the raw corpus file or a DataFrame with the corpus.
        corpus_name : str
            Logical corpus name.
        metadata : dict | None
            Additional corpus metadata. NaN/Inf will be converted to None for JSON safety.
        logger : logging.Logger | None
            Logger to use; if None, a module-level logger is created.
        """

        # Logger
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level="INFO")
            self._logger = logging.getLogger("Entity Corpus")

        # Load/validate data
        if data is None:
            raise ValueError("No data provided for corpus.")

        if isinstance(data, str):
            path = pathlib.Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Path to raw data {path} does not exist.")
            df = read_dataframe(path)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise TypeError("`data` must be a str or a pandas.DataFrame.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Loaded data is not a pandas DataFrame.")

        # Keep the original data but compute fields once
        self.data: pd.DataFrame = df
        self.corpus_name: str = str(corpus_name)
        self.fields: List[str] = [str(c) for c in self.data.columns.tolist()]

        # Sanitize metadata for JSON safety and avoid mutable defaults
        meta = dict(metadata) if metadata else {}
        self.metadata: dict = {k: _json_safe(v) for k, v in meta.items()}

        if self.data.empty:
            self._logger.warning("Initialized with an empty DataFrame.")

    def get_corpus_info(self) -> Iterable[dict]:
        """
        Yields one record at a time as dictionaries, without materializing the
        entire DataFrame as a list (memory-efficient).
        """
        cols = self.fields
        for row in self.data.itertuples(index=False, name=None):
            yield {k: _json_safe(v) for k, v in zip(cols, row)}

    def get_corpora_update(self, id: int) -> List[dict]:
        """
        Creates the JSON payload to update the 'corpora' collection in Solr with
        the new logical corpus information. Ensures all values are JSON-safe.
        """
        return [
            {
                "id": id,
                "corpus_name": self.corpus_name,
                "fields": [str(f) for f in self.fields],
                **{k: _json_safe(v) for k, v in self.metadata.items()},
            }
        ]