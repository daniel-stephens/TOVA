import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
import re

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

model = SentenceTransformer('paraphrase-distilroberta-base-v2')

def preprocess_file(file_path, file_type):
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

    df = df.dropna(subset=['Content'])

    texts = df['Content'].tolist()

    docs = list(nlp.pipe(texts))

    # TF-IDF filtering
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    low_value_words = set([word for word, idf in zip(vectorizer.get_feature_names_out(), vectorizer.idf_) if idf <= 3])

    # Expanded stop words
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

    embeddings = model.encode(processed_texts, batch_size=32, show_progress_bar=True)

    df['Processed_Content'] = processed_texts
    df['Embeddings'] = embeddings.tolist()

    return df
