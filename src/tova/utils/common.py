import json
import logging
import os
import pathlib
import pickle
from datetime import datetime
import sys
from typing import Any, Dict, Optional

import yaml

class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def log_or_print(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Helper function to log or print messages.

    Parameters
    ----------
    message : str
        The message to log or print.
    level : str, optional
        The logging level, by default "info".
    logger : logging.Logger, optional
        The logger to use for logging, by default None.
    """
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
    else:
        print(message)


def load_yaml_config_file(
    config_file: str,
    section: str,
    logger:logging.Logger
) -> Dict:
    """
    Load a YAML configuration file and return the specified section.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.
    section : str
        Section of the configuration file to return.

    Returns
    -------
    Dict
        The specified section of the configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If the specified section is not found in the configuration file.
    """

    if not pathlib.Path(config_file).exists():
        log_or_print(f"Config file not found: {config_file}", level="error", logger=logger)
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    section_dict = config.get(section, {})

    if section == {}:
        log_or_print(f"Section {section} not found in config file.", level="error", logger=logger)
        raise ValueError(f"Section {section} not found in config file.")

    log_or_print(f"Loaded config file {config_file} and section {section}.", logger=logger)

    return section_dict

def init_logger(
    config_file: str,
    name: str = None
) -> logging.Logger:
    """
    Initialize a logger based on the provided configuration.

    Parameters
    ----------
    config_file : str
        The path to the configuration file.
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """

    logger_config = load_yaml_config_file(config_file, "logger", logger=None)
    name = name if name else logger_config.get("logger_name", "default_logger")
    log_level = logger_config.get("log_level", "INFO").upper()
    dir_logger = pathlib.Path(logger_config.get("dir_logger", "logs"))
    N_log_keep = int(logger_config.get("N_log_keep", 5))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create path_logs dir if it does not exist
    dir_logger.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved in {dir_logger}")

    # Generate log file name based on the data
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{name}_log_{current_date}.log"
    log_file_path = dir_logger / log_file_name

    # Remove old log files if they exceed the limit
    log_files = sorted(
        dir_logger.glob("*.log"),
        key=lambda f: f.stat().st_mtime, reverse=True)
    if len(log_files) >= N_log_keep:
        for old_file in log_files[N_log_keep - 1:]:
            old_file.unlink()

    # Create handlers based on config
    if logger_config.get("file_log", True):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    if logger_config.get("console_log", True):
        #console_handler = logging.StreamHandler()
        console_handler = FlushingStreamHandler(sys.stdout)

        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger

def unpickler(file: str) -> object:
    """
    Unpickle file

    Parameters
    ----------
    file : str
        The path to the file to unpickle.

    Returns
    -------
    object
        The unpickled object.
    """
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob: object) -> int:
    """
    Pickle object to file

    Parameters
    ----------
    file : str
        The path to the file where the object will be pickled.
    ob : object
        The object to pickle.

    Returns
    -------
    int
        0 if the operation is successful.
    """
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return 0

def get_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with an optional prefix.

    Parameters
    ----------
    prefix : str, optional
        The prefix to add to the unique ID, by default "".

    Returns
    -------
    str
        The generated unique ID.
    """
    import uuid
    return f"{prefix}{uuid.uuid4().hex}"


def write_json_atomic(path: pathlib.Path, payload: Any) -> None:
    """Safely write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    
    
def pydantic_to_dict(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    # handle list-like passed by mistake
    if isinstance(model, (list, tuple)):
        return {"items": [pydantic_to_dict(m) if not isinstance(m, dict) else m for m in model]}

    dump_method = getattr(model, "model_dump", None)
    if callable(dump_method):
        return dump_method(by_alias=True)
    dict_method = getattr(model, "dict", None)
    if callable(dict_method):
        return dict_method(by_alias=True)

    try:
        return dict(model)
    except Exception:
        try:
            import json
            return json.loads(json.dumps(model, default=lambda o: getattr(o, "__dict__", {})))
        except Exception:
            return {}

