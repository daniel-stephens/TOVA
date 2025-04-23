import logging
from typing import Optional
from .common import log_or_print
import numpy as np
import pathlib
import pandas as pd

def file_lines(fname: pathlib.Path) -> int:
    """
    Count number of lines in file

    Parameters
    ----------
    fname: pathlib.Path
        The file whose number of lines is calculated.

    Returns
    -------
    int
        Number of lines in the file.
    """
    with fname.open('r', encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_dataframe(
    path_to_data: pathlib.Path,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Read a dataframe from a file.
    Supported file formats: parquet, json, jsonl.

    Parameters
    ----------
    path_to_data : pathlib.Path
        Path to the file containing the data.
    logger : logging.Logger, optional
        Logger for logging messages. Defaults to None.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the data.

    Raises
    ------
    ValueError
        If the file format is unsupported.
    RuntimeError
        If an error occurs while reading the data.
    """
    try:
        log_or_print(
            f"Loading data from {path_to_data}...", logger=logger)

        # Determine file format based on the suffix
        if path_to_data.suffix == ".parquet":
            df = pd.read_parquet(path_to_data)
        elif path_to_data.suffix in {".json", ".jsonl"}:
            df = pd.read_json(path_to_data, lines=True)
        elif path_to_data.suffix == ".csv":
            df = pd.read_csv(path_to_data)
        else:
            err_msg = f"Unsupported file format: {path_to_data.suffix}"
            log_or_print(err_msg, level="error", logger=logger)
            raise ValueError(err_msg)

        log_or_print(f"Data successfully loaded. Shape: {df.shape}")
        return df

    except Exception as e:
        err_msg = f"An error occurred while reading the data: {e}"
        log_or_print(err_msg, level="error", logger=logger)
        raise RuntimeError(err_msg)
    
    
def get_embeddings_from_str(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Get embeddings from a DataFrame, assuming there is a column named 'embeddings' with the embeddings as strings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the embeddings as strings in a column named 'embeddings'.
    logger : Union[logging.Logger, None], optional
        Logger for logging errors, by default None.

    Returns
    -------
    np.ndarray
        Array of embeddings.

    Raises
    ------
    KeyError
        If the 'embeddings' column is in the df.
    ValueError
        If the 'embeddings' column contains invalid data that cannot be converted to embeddings.
    """

    # Check if 'embeddings' column exists
    if "embeddings" not in df.columns:
        err_msg = "DataFrame does not contain 'embeddings' column."
        log_or_print(err_msg, level="error", logger=logger)
        raise KeyError(err_msg)

    # Extract embeddings
    embeddings = df.embeddings.values.tolist()

    # Check if embeddings are in string format and convert
    try:
        if isinstance(embeddings[0], str):
            embeddings = np.array(
                [np.array(el.split(), dtype=np.float32) for el in embeddings]
            )
        else:
            raise ValueError(
                "Embeddings are not in the expected string format.")
    except Exception as e:
        err_msg = f"Error processing embeddings: {e}"
        log_or_print(err_msg, level="error", logger=logger)
        raise ValueError(err_msg)

    return np.array(embeddings)