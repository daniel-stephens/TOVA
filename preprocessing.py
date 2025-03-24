import argparse
import json
import pickle
import re
import torch
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
import os
import json
import PyPDF2
from werkzeug.utils import secure_filename
from io import BytesIO

def load_data(files):
    """Loads data from supported formats: JSON, JSONL, CSV, Excel, and PDF from Flask uploads."""
    data = []

    for file in files:
        file_ext = file.filename.split('.')[-1].lower()
        file_stream = BytesIO(file.read())  # Convert uploaded file to a stream

        if file_ext in ['json', 'jsonl']:
            file_stream.seek(0)  # Reset stream position
            for line in file_stream:
                data.append(json.loads(line) if file_ext == 'jsonl' else json.load(file_stream))

        elif file_ext in ['csv', 'xlsx']:
            file_stream.seek(0)
            df = pd.read_csv(file_stream) if file_ext == 'csv' else pd.read_excel(file_stream)
            if {'Document Number', 'Content'}.issubset(df.columns):
                data.extend(df.to_dict(orient='records'))
            else:
                raise ValueError("Missing required fields: 'Document Number' and 'Content'")

        elif file_ext == 'pdf':
            file_stream.seek(0)
            reader = PyPDF2.PdfReader(file_stream)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() if page.extract_text() else ""
                data.append({"Document Number": f"{secure_filename(file.filename)}_Page_{i+1}", "Content": text})

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    return json.dumps(data, ensure_ascii=False, indent=4)





def concatenate_text_columns(df, text_columns):
    """Concatenates multiple text columns into a single string."""
    text_columns = text_columns.split(",") if isinstance(text_columns, str) else [text_columns]
    df = df.dropna(subset=text_columns, how="any")
    return df.apply(lambda row: ' '.join(str(row[col]) for col in text_columns), axis=1)

def get_filtered_words(text, threshold=0):
    """Uses TF-IDF to find low-importance words."""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)
    return [word for word, idf in zip(vectorizer.get_feature_names_out(), vectorizer.idf_) if idf <= threshold]

def tokenize_and_clean_text(data, data_path):
    """Tokenizes, removes stopwords, and filters text data."""
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')
    docs = [nlp(doc) for doc in tqdm(data, desc="Tokenizing documents")]
    stop_words = STOP_WORDS.union(get_filtered_words(data))
    data_words_nonstop, word_spans = [], []
    for doc in docs:
        temp_doc, temp_span = [], []
        for token in doc:
            if re.search('[a-z0-9]+', str(token)) and not token.is_digit and not token.is_space \
                    and str(token).lower() not in stop_words and str(token).strip() != "":
                temp_doc.append(token.lemma_.lower())
                temp_span.append((token.idx, token.idx + len(token)))
        data_words_nonstop.append(temp_doc)
        word_spans.append(temp_span)
    return data_words_nonstop, word_spans

def calculate_embedding(df, batch_size, sbert_model):
    """Computes sentence embeddings using SBERT."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(sbert_model).to(device)
    df["embeddings"] = df["calculate_on"].apply(lambda text: ' '.join(str(x) for x in model.encode(text, show_progress_bar=True, batch_size=batch_size)))
    return df

def save_data(data_words_nonstop, word_spans, texts, labels, save_path):
    """Saves processed data to a Pickle file."""
    with open(save_path, 'wb+') as outp:
        pickle.dump({'datawords_nonstop': data_words_nonstop, 'spans': word_spans, 'texts': texts, 'labels': labels}, outp)

def dump_new_json(texts, data_words_nonstop, embeddings, labels, sub_labels, save_path):
    """Saves the cleaned dataset in JSON format, removing short documents."""
    result = {"text": {}, "label": {}, "sub_labels": {}, "processed_text": {}, "embeddings": {}}
    counter = 0
    for i in range(len(texts)):
        if len(texts[i].split()) > 10 and len(data_words_nonstop[i]) > 4:
            result['processed_text'][str(counter)] = ' '.join(data_words_nonstop[i])
            result['text'][str(counter)] = texts[i]
            result['embeddings'][str(counter)] = embeddings[i]
            result['label'][str(counter)] = labels[i]
            result['sub_labels'][str(counter)] = sub_labels[i]
            counter += 1
    with open(save_path, "w") as outfile:
        json.dump(result, outfile)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", type=str, required=True)
    argparser.add_argument("--save_path", type=str, required=True)
    argparser.add_argument("--new_json_path", type=str, required=True)
    argparser.add_argument("--text_columns", type=str, default="text")
    argparser.add_argument("--label_column", type=str, default="label")
    argparser.add_argument("--sub_label_column", type=str, default="sub_labels")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--sbert_model", type=str, default="paraphrase-distilroberta-base-v2")
    args = argparser.parse_args()

    df = load_data(args.doc_dir)
    df["calculate_on"] = concatenate_text_columns(df, args.text_columns)
    texts, labels, sub_labels = df["calculate_on"].tolist(), df[args.label_column].tolist(), df[args.sub_label_column].tolist()
    data_words_nonstop, word_spans = tokenize_and_clean_text(texts, args.doc_dir)
    df = calculate_embedding(df, args.batch_size, args.sbert_model)
    embeddings = df.embeddings.tolist()
    save_data(data_words_nonstop, word_spans, texts, labels, args.save_path)
    dump_new_json(texts, data_words_nonstop, embeddings, labels, sub_labels, args.new_json_path)

if __name__ == "__main__":
    main()
