# top2vec_adapter.py
from top2vec import Top2Vec
from adapters.base_model import BaseTopicModel


class Top2VecAdapter(BaseTopicModel):
    def __init__(self):
        self.model = None

    def fit(self, documents, embeddings=None):
        self.model = Top2Vec(documents, speed="learn", workers=4)

    def get_topics(self):
        topic_words, _, _ = self.model.get_topics()
        return {i: words for i, words in enumerate(topic_words)}

    def get_document_topics(self):
        doc_topics, _, _ = self.model.get_documents_topics(range(len(self.model.documents)))
        return {i: topic for i, topic in enumerate(doc_topics)}
