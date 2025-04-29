import os
import re
import pandas as pd
import spacy
from datetime import datetime
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import (
    TextLoader, CSVLoader, UnstructuredFileLoader, PDFMinerLoader
)
from langchain.schema import Document as LCDocument

# Load models and NLP pipeline
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')
embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v2')

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
    texts = df['Content'].tolist()
    docs = list(nlp.pipe(texts))

    # TF-IDF filtering
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    low_value_words = set([
        word for word, idf in zip(vectorizer.get_feature_names_out(), vectorizer.idf_) if idf <= 3
    ])
    stop_words = STOP_WORDS.union(low_value_words)

    processed_texts = []
    for doc in docs:
        tokens = [token.lemma_.lower() for token in doc if
                  re.search('[a-z0-9]+', token.text) and
                  len(token.text) > 1 and
                  not token.is_digit and
                  not token.is_space and
                  token.lemma_.lower() not in stop_words]

        cleaned_tokens = [''.join([char for char in tok if char.isalpha()]) for tok in tokens]
        cleaned_tokens = [tok for tok in cleaned_tokens if tok.strip() != '']
        processed_texts.append(' '.join(cleaned_tokens))

    # embeddings = embedding_model.encode(processed_texts, batch_size=32, show_progress_bar=True)

    output = []
    for i, row in df.iterrows():
        metadata = row.to_dict()
        del metadata['Content']
        output.append({
            "content": processed_texts[i],
            # "embedding": embeddings[i].tolist(),
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
            "embedding": row["embedding"],
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

def preprocess_and_embed(
    file_path: str,
    file_type: str,
    text_columns: list,
    sbert_model: str = "paraphrase-distilroberta-base-v2",
    tfidf_threshold: float = 3,
    batch_size: int = 32,
    min_tokens: int = 5
) -> pd.DataFrame:
    """
    Load file, preprocess texts, and compute embeddings for database insertion.

    Args:
        file_path (str): Path to the file.
        file_type (str): Type of file (csv, json, jsonl, xlsx, xls, txt).
        text_columns (list): Text columns to process.
        label_column (str): Optional label/category column.
        sbert_model (str): SentenceTransformer model.
        tfidf_threshold (float): Threshold for tf-idf stopword expansion.
        batch_size (int): Batch size for embeddings.
        min_tokens (int): Minimum tokens for accepting preprocessed text.

    Returns:
        pd.DataFrame: Processed dataframe with processed_text and embedding.
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
        df = pd.DataFrame({"Document Number": [1], "Content": [content]})
    else:
        raise ValueError("Unsupported file type")

    # --- Drop rows with missing text columns ---
    df = df.dropna(subset=text_columns, how="any")

    # --- Calculate text to process ---
    df["calculate_on"] = df.apply(lambda row: ' '.join([str(row[col]) for col in text_columns]), axis=1)
    texts = df["calculate_on"].tolist()

    # --- NLP Preprocessing ---
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("sentencizer")
    docs = list(nlp.pipe(texts))

    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    tfidf_stopwords = set([
        word for word, idf in zip(vectorizer.get_feature_names_out(), vectorizer.idf_)
        if idf <= tfidf_threshold
    ])
    stop_words = STOP_WORDS.union(tfidf_stopwords)

    processed_texts = []
    for i, doc in enumerate(docs):
        tokens = [
            token.lemma_.lower() for token in doc
            if re.search('[a-z0-9]+', token.text)
            and len(token.text) > 1
            and not token.is_digit
            and not token.is_space
            and token.lemma_.lower() not in stop_words
        ]
        cleaned_tokens = [''.join([c for c in t if c.isalpha()]) for t in tokens]
        cleaned_tokens = [tok for tok in cleaned_tokens if tok.strip()]

        if len(cleaned_tokens) < min_tokens:
            processed_texts.append(df["calculate_on"].iloc[i])
        else:
            processed_texts.append(' '.join(cleaned_tokens))

    # --- Embedding ---
    # print("🚩 Calculating embeddings...")
    # model = SentenceTransformer(sbert_model)
    # embeddings = model.encode(processed_texts, batch_size=batch_size, show_progress_bar=True)

    # --- Attach Results ---
    df["processed_text"] = processed_texts
    # df["embedding"] = [embedding.tolist() for embedding in embeddings]

    print("✅ Preprocessing completed!")
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


