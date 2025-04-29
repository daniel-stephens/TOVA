import os
import chromadb
import pandas as pd
# from adapters.lda_adapter import LDAAdapter

def list_models():
    return [f.replace(".model", "") for f in sorted(os.listdir("model")) if f.endswith(".model")]

def list_datasets(model_name):
    return ["documents"]  # Can be expanded

def load_topic_model(model_name, num_topics=10):
    return LDAAdapter.load(f"model/{model_name}")

def get_model_documents_and_topics(client, collection_name, model):
    collection = client.get_or_create_collection(name=collection_name)
    data = collection.get(include=["documents", "metadatas"])
    documents = data["documents"]

    tokenized_docs = [doc.split() for doc in documents]
    bows = [model.dictionary.doc2bow(doc) for doc in tokenized_docs]
    model.documents = tokenized_docs
    model.corpus = bows
    topics = model.get_document_topics()

    return documents, topics
