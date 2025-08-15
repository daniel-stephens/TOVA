import json
import logging
from typing import Optional
from tova.utils.common import log_or_print
import numpy as np
from pathlib import Path
import pandas as pd # type: ignore
from typing import List, Dict, Any, Optional

def file_lines(fname: Path) -> int:
    """
    Count number of lines in file

    Parameters
    ----------
    fname: Path
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
    path_to_data: Path,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Read a dataframe from a file.
    Supported file formats: parquet, json, jsonl.

    Parameters
    ----------
    path_to_data : Path
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

def prepare_training_data(
    path: str,
    logger,
    text_col: str = "tokenized_text",
    id_col: Optional[str] = "id",
    get_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    path = Path(path)
    df = read_dataframe(path, logger)

    if get_embeddings:
        if "embeddings" in df.columns:
            df["embeddings"] = get_embeddings_from_str(df, logger)
            logger.info("Embeddings loaded from DataFrame.")
        else:
            logger.warning("Embeddings column not found — embeddings will be omitted.")
            df["embeddings"] = None

    if id_col:
        if id_col in df.columns:
            df.rename(columns={id_col: "id"}, inplace=True)
        else:
            df["id"] = range(1, len(df) + 1)

    if text_col in df.columns:
        df.rename(columns={text_col: "raw_text"}, inplace=True)
    else:
        logger.warning(f"Text column '{text_col}' not found in DataFrame.")
        raise ValueError(f"Text column '{text_col}' not found in DataFrame.")

    allowed_columns = ["id", "raw_text", "embeddings"]
    df = df[[col for col in df.columns if col in allowed_columns]]

    return df.to_dict(orient="records")

def normalize_json_data(
    raw_data: str,
    logger: logging.Logger,
    id_col: Optional[str] = "id",
    text_col: str = "tokenized_text",
) -> list[dict]:
    """
    Normalize raw JSON input to match prepare_training_data() output.
    Ensures 'id' and 'raw_text' fields are set.
    """
    data = json.loads(raw_data)

    logger.info(f"Normalizing JSON data: {len(data)} records found.")
    logger.info(f"Converting {id_col} to 'id' and {text_col} to 'raw_text'.")
    
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of records.")

    for i, row in enumerate(data):
        if id_col and id_col in row:
            row["id"] = row[id_col]
        elif "id" not in row:
            row["id"] = i + 1

        if text_col not in row:
            raise ValueError(f"Missing expected text column '{text_col}' in row {i}")

        row["raw_text"] = row[text_col]
    
    logger.info(f"Normalization complete: {len(data)} records processed.")

    return data
