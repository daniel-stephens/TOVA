from gensim import corpora, models
from .base_model import BaseTopicModel


class LDAAdapter(BaseTopicModel):
    def __init__(self, num_topics=10, passes=10, random_state=42):
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.documents = []

    def fit(self, documents, embeddings=None):
        self.documents = [doc.split() for doc in documents]
        self.dictionary = corpora.Dictionary(self.documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        self.model = models.LdaModel(
            self.corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=self.passes,
            random_state=self.random_state
        )

    def get_topics(self):
        return {
            topic_id: [word for word, _ in self.model.show_topic(topic_id, topn=10)]
            for topic_id in range(self.num_topics)
        }

    def get_document_topics(self):
        doc_topics = {}

        for i, bow in enumerate(self.corpus):
            dist = self.model.get_document_topics(bow, minimum_probability=0.0)
            dominant_topic, score = max(dist, key=lambda x: x[1])
            doc_topics[i] = {
                "topic_id": dominant_topic,
                "topic_score": float(score),
                "topic_distribution": {tid: float(prob) for tid, prob in dist}
            }

        return doc_topics

    def get_topic_scores(self):
        # Optional: Return frequency of each topic
        topic_freq = [0] * self.num_topics
        for bow in self.corpus:
            for tid, prob in self.model.get_document_topics(bow):
                topic_freq[tid] += prob
        return {i: float(score) for i, score in enumerate(topic_freq)}

    def get_topic_embeddings(self):
        # Gensim LDA does not support topic embeddings
        return None

    def get_model_name(self):
        return "LDA"

    def save(self, filepath):
        self.model.save(filepath + ".model")
        self.dictionary.save(filepath + ".dict")

    @classmethod
    def load(cls, filepath):
        instance = cls()
        instance.model = models.LdaModel.load(filepath + ".model")
        instance.dictionary = corpora.Dictionary.load(filepath + ".dict")
        return instance
