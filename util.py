import pandas as pd
from datetime import datetime
from langchain.document_loaders import (
    TextLoader, CSVLoader)
# from celery import Celery
# import redis



def preprocess_file(file_path, file_type, label_column):
    if file_type == 'json':
        df = pd.read_json(file_path)
    elif file_type == 'jsonl':
        df = pd.read_json(file_path, lines=True)
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type in ['excel', 'xlsx', 'xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")

    required_columns = ['Document Number', 'Content']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=[label_column])

    output = []
    for _, row in df.iterrows():
        metadata = row.to_dict()
        content = metadata.pop('Content')
        output.append({
            "content": content,
            "metadata": metadata
        })

    return output


def excel_confirmation(file, text_column, label_column=None) -> dict:
    try:
        df = pd.read_excel(file)

        # Check if the required text column exists
        if text_column not in df.columns:
            return {
                "status": "error",
                "message": f"Validation error: Required column '{text_column}' is missing."
            }

        # If label_column is provided, check it too
        if label_column and label_column not in df.columns:
            return {
                "status": "error",
                "message": f"Validation error: Label column '{label_column}' is missing."
            }

        return {
            "status": "success",
            "message": f"Validation complete: '{text_column}' column"
                       f"{' and ' + label_column if label_column else ''} found."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Validation failed: {str(e)}"
        }

def load_file(file_path):
    ext = file_path.split('.')[-1].lower()
    docs = []

    if ext == 'txt':
        loader = TextLoader(file_path)
        docs = loader.load()

    elif ext == 'csv':
        loader = CSVLoader(file_path)
        docs = loader.load()

    elif ext in ['json', 'jsonl', 'xls', 'xlsx']:  # 🔁 Use your custom processor (which uses pandas)
        preprocessed = preprocess_file(file_path, ext)
        return [{
            "content": row["content"],
            "metadata": {
                **row["metadata"],
                "source": file_path,
                "created_at": datetime.utcnow().isoformat()
            }
        } for row in preprocessed]

    else:
        raise ValueError(f"Unsupported file extension: .{ext}")

    contents = [doc.page_content for doc in docs]
    # embeddings = embedding_model.encode(contents, batch_size=32)

    return [{
        "content": contents[i],
        # "embedding": embeddings[i].tolist(),
        "metadata": {
            **docs[i].metadata,
            "source": file_path,
            "created_at": datetime.utcnow().isoformat()
        }
    } for i in range(len(docs))]


import json

def parse_multiline_key_value_string(text, to_json=False):
    """
    Parses a string with key-value pairs separated by newlines into a dictionary.
    
    Args:
        text (str): String with format 'Key: Value\\nKey: Value\\n...'
        to_json (bool): If True, returns a JSON string instead of a dict.
        
    Returns:
        dict or str: Parsed data as a dict or JSON string.
    """
    lines = text.strip().split('\n')
    data = {}

    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()

    return json.dumps(data, indent=2) if to_json else data


def processFile(
    file_path: str,
    file_type: str,
    text_column: str
) -> pd.DataFrame:
    """
    Load file and extract a single specified text column for database insertion.

    Args:
        file_path (str): Path to the file.
        file_type (str): Type of file (csv, json, jsonl, xlsx, xls, txt).
        text_column (str): The name of the text column to extract.

    Returns:
        pd.DataFrame: DataFrame with a single 'Context' column.
    """

    # --- Load File ---
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'json':
        df = pd.read_json(file_path)
    elif file_type == 'jsonl':
        df = pd.read_json(file_path, lines=True)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(file_path)
    elif file_type == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return pd.DataFrame({"Context": [content]})
    else:
        raise ValueError("Unsupported file type")

    # --- Drop rows with missing values in the text column ---
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' column not found in file")

    df = df.dropna(subset=[text_column])  # Wrap in list

    # --- Rename column to 'Context' and return as DataFrame ---
    df = df[[text_column]].rename(columns={text_column: "Context"})

    return df.reset_index(drop=True)


def get_corpus_data(corpus_name, coll):
    """
    Retrieve all documents, metadata, and ids for a given corpus name.
    """
    results = coll.get(
        where={"corpus_name": corpus_name},  # 👈 Filter by corpus name
        include=["documents", "metadatas"],  # 👈 Only need documents and metadatas
        limit=10000  # Adjust if you expect very large corpora
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])  # ✅ ids are automatically returned even without include

    return documents, metadatas, ids
