import pandas as pd
from datetime import datetime
from langchain.document_loaders import (
    TextLoader, CSVLoader)
import chromadb
client = chromadb.PersistentClient(path="database/myDB")
collection = client.get_or_create_collection(name="documents")
registry = client.get_or_create_collection("corpus_model_registry")


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


import requests

def fetch_and_process_model_info(model_path: str, endpoint: str = "http://127.0.0.1:8989/queries/model-info"):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "config_path": "static/config/config.yaml",
        "model_path": f"models/{model_path}"
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    print("Status Code:", response.status_code)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    
    allInfo = response.json()

    themeDetails = build_theme_data_dict(allInfo, collection)
    summary = extract_topic_summaries(allInfo)
    themeSummary = sorted(summary, key=lambda t: t["document_count"], reverse=True)
    return themeSummary, themeDetails


def call_gateway(url, method="POST", payload=None, headers=None):
    """
    Call a gateway route with GET or POST.

    Args:
        url (str): The gateway endpoint URL.
        method (str): HTTP method ("GET" or "POST").
        params (dict, optional): Query parameters for GET.
        payload (dict, optional): JSON body for POST.
        headers (dict, optional): Headers to include.

    Returns:
        dict: JSON response if successful.
        str: Error message if request fails.
    """
    try:
        method = method.upper()
        if method == "GET":
            response = requests.get(url, params=["GET, POST"], headers=headers)
        elif method == "POST":
            headers = headers or {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
        else:
            return f"Unsupported method: {method}"

        if response.ok:
            return response.json()
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"


def build_theme_data_dict(allInfo, doc_collection):
    """
    Builds a dictionary of theme data indexed by ID, with text and score for each assigned document.
    """
    theme_dict = {}
    topics_info = allInfo.get("Topics Info", {})
    topic_keys = list(topics_info.keys())

    for idx, topic_key in enumerate(topic_keys):
        topic_data = topics_info[topic_key]
        theme_id = topic_key

        # Extract top_doc text
        top_doc_text = ""
        top_docs = topic_data.get("Top Documents", {})
        if isinstance(top_docs, dict) and top_docs:
            first_doc_key = next(iter(top_docs))
            top_doc_text = top_docs[first_doc_key]

        # Assigned Documents with Scores
        docs_prob = topic_data.get("Assigned Documents", {})
        assigned_results = []
        if docs_prob:
            ids = list(docs_prob.keys())
            results = doc_collection.get(ids=ids, include=["documents"])
            assigned_results = [
                {
                    "id": doc_id,
                    "text": text,
                    "score": docs_prob.get(doc_id, 0.0),
                    "theme": topic_data.get("Label", f"Topic {idx}"),
                }
                for doc_id, text in zip(results["ids"], results["documents"])
            ]

        

        # Build theme data
        theme_data = {
            "id": theme_id,
            "label": topic_data.get("Label", f"Topic {idx}"),
            "prevalence": topic_data["Size"],
            "coherence": topic_data["Coherence (NPMI)"],
            "entropy": topic_data["Entropy"],
            "size" : topic_data["Size"],
            "keywords": topic_data["Keywords"].split(", "),
            "summary": topic_data["Summary"],
            "top_doc": top_doc_text,
            "theme_matches": len(docs_prob),
            "Coordinates": topic_data["Coordinates"],
            "similar_themes": topic_data.get("Similar Topics (Coocurring)", []),
            "trend": [],  # Add if you compute this
            "documents": assigned_results
        }

        theme_dict[theme_id] = theme_data

    return theme_dict


def get_assigned_documents_with_scores(doc_collection, docs_prob):
    """
    Given a ChromaDB document collection and a dictionary of Assigned Documents
    (with scores), return a list of {id, text, score} for each document.
    """
    if not docs_prob:
        return []

    doc_ids = list(docs_prob.keys())

    # Fetch document texts from ChromaDB
    results = doc_collection.get(
        ids=doc_ids,
        include=["documents"]
    )

    return [
        {
            "id": doc_id,
            "text": text,
            "score": docs_prob.get(doc_id, 0.0)
        }
        for doc_id, text in zip(results["ids"], results["documents"])
    ]

def extract_topic_summaries(allInfo):
    topic_info = allInfo.get("Topics Info", {})
    topic_summaries = []

    for idx, (topic_key, topic_data) in enumerate(topic_info.items()):
        label = topic_data.get("Label", f"Topic {idx}")
        document_count = len(topic_data.get("Assigned Documents", []))

        topic_summaries.append({
            "id": topic_key,
            "label": label,
            "document_count": document_count
        })

    return topic_summaries



def load_or_create_dashboard_json(path: str = "static/config/dashboardData.json") -> dict:
    """
    Loads existing dashboard JSON data if available,
    or creates an empty JSON file and returns an empty dict.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: The dashboard data.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        # Load existing data
        with file_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                print(f"📂 Loaded dashboard data from {file_path.resolve()}")
            except json.JSONDecodeError:
                print(f"⚠️ File was invalid. Resetting to empty JSON.")
                data = {}
    else:
        # Create empty file
        data = {}
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"🆕 Created empty dashboard JSON at {file_path.resolve()}")

    return data


def add_model_to_dashboard(model_name: str, themeSummary, themeDetails, path: str = "static/config/dashboardData.json") -> dict:
    """
    Adds a new model entry to an existing dashboard JSON file.

    Args:
        model_name (str): The name of the model to add.
        path (str): Path to the JSON file.

    Returns:
        dict: The updated dashboard data.
    """
    file_path = Path(path)
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Add new model entry
    data[model_name] = {
        "Theme Summary": themeSummary,
        "Theme Details": themeDetails
    }

    # Write back to file
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Added model '{model_name}' to dashboard JSON")
    # return data

import json
from pathlib import Path

def read_dashboard_json(path: str = "static/config/dashboardData.json") -> dict:
    """
    Reads the dashboardData.json file and returns its contents as a dictionary.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: The dashboard data. Returns an empty dict if the file doesn't exist or is invalid.
    """
    file_path = Path(path)

    if not file_path.exists():
        print(f"⚠️ File not found at {file_path.resolve()}. Returning empty dict.")
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"📂 Successfully read dashboard data from {file_path.resolve()}")
            return data
    except json.JSONDecodeError:
        print(f"❌ JSON format error in {file_path.resolve()}. Returning empty dict.")
        return {}


def extract_metrics_from_theme_details(theme_details: dict) -> list:
    metrics = []

    for topic_id, topic_data in theme_details.items():
        try:
            prevalence_str = topic_data.get("prevalence", "0%").replace("%", "")
            prevalence = round(float(prevalence_str) / 100, 3)  # Convert "18.44%" → 0.184

            metrics.append({
                "theme": topic_data.get("label", topic_id),  # 👈 now uses 'theme'
                "prevalence": prevalence,
                "coherence": topic_data.get("coherence"),
                "enthropy": topic_data.get("entropy")
            })
        except Exception as e:
            print(f"Skipping topic {topic_id} due to error: {e}")

    return metrics

import requests

def infer_text_(id, raw_text, config_path="static/config/config.yaml", url="http://127.0.0.1:8989/infer/json"):
    payload = {
        "config_path": config_path,
        "data": [
            {
                "id": id,
                "raw_text": raw_text
            }
        ]
    }

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.RequestException as e:
        print("Request failed:", e)
        return None



def get_thetas_by_doc_ids( doc_id, model):
    url = 'http://127.0.0.1:8989/queries/thetas-by-docs-ids'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    payload = {
        "config_path": "static/config/config.yaml",
        "docs_ids": doc_id,
        "model_path": f"models/{model}"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return None


def format_theta_output_dict(thetas, theme_summary, rationale=None):
    # Create a mapping from topic ID to label
    label_lookup = {
        summary["id"].replace("t", ""): summary["label"]
        for summary in theme_summary
    }

    results = {}

    for doc_id, probs in thetas.items():
        # Sort the topic probabilities in descending order
        sorted_topics = sorted(probs.items(), key=lambda x: -x[1])

        # Build top themes with readable labels, IDs, and rounded scores
        top_themes = [
            {
                "theme_id": f"t{tid}",
                "label": label_lookup.get(tid, f"Theme {tid}"),
                "score": round(score, 2)
            }
            for tid, score in sorted_topics
        ]

        # Construct the output dict for this document
        doc_info = {
            "theme": top_themes[0]["label"],
            "top_themes": top_themes
        }

        if rationale:
            doc_info["rationale"] = rationale

        results[doc_id] = doc_info

    return results

