# bertopic_adapter.py
from bertopic import BERTopic
from adapters.base_model import BaseTopicModel


class BERTopicAdapter(BaseTopicModel):
    def __init__(self):
        self.model = BERTopic()
        self.documents = None  # <-- Add this

    def fit(self, documents, embeddings=None):
        self.documents = documents  # <-- Store documents for later use
        if embeddings:
            self.model.fit(documents, embeddings)
        else:
            self.model.fit(documents)

    def get_topics(self):
        return {topic: words for topic, words in self.model.get_topics().items()}

    def get_document_topics(self):
        doc_topics, _ = self.model.transform(self.documents)  # <-- Use self.documents
        return {i: topic for i, topic in enumerate(doc_topics)}
