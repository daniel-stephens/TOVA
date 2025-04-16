class BaseTopicModel:
    """
    Abstract base class for all topic models integrated into the system.
    Every custom model (LDA, BERTopic, Top2Vec, etc.) should inherit from this class
    and implement the required methods to be compatible with the system.
    """

    def fit(self, documents, embeddings=None):
        raise NotImplementedError("fit() must be implemented.")

    def get_topics(self):
        raise NotImplementedError("get_topics() must be implemented.")

    def get_document_topics(self):
        raise NotImplementedError("get_document_topics() must be implemented.")

    def get_topic_embeddings(self):
        return None

    def get_topic_scores(self):
        return None

    def get_model_name(self):
        return self.__class__.__name__

    def transform(self, documents, embeddings=None):
        raise NotImplementedError("transform() is not implemented.")

    @property
    def requires_embeddings(self):
        return False

    def save(self, filepath):
        """
        Save the trained topic model to disk.

        Args:
            filepath (str): File path where the model should be saved.
        """
        raise NotImplementedError("save() must be implemented by subclasses.")

    @classmethod
    def load(cls, filepath):
        """
        Load a saved topic model from disk.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            BaseTopicModel: An instance of the model.
        """
        raise NotImplementedError("load() must be implemented by subclasses.")
