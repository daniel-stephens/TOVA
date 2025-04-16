class BaseTopicModel:
    """
    Abstract base class for all topic models integrated into the system.
    Every custom model (LDA, BERTopic, Top2Vec, etc.) should inherit from this class
    and implement the required methods to be compatible with the system.
    """

    def fit(self, documents, embeddings=None):
        """
        Train the model on a list of documents.

        Args:
            documents (List[str]): Preprocessed input documents.
            embeddings (Optional[List[List[float]]]): Optional dense vector embeddings 
                                                      (used in models like BERTopic or Top2Vec).
        """
        raise NotImplementedError("fit() must be implemented.")

    def get_topics(self):
        """
        Get the discovered topics.

        Returns:
            dict: A dictionary where keys are topic IDs and values are lists of top words.
                  Example: {0: ["economy", "growth", "inflation"], ...}
        """
        raise NotImplementedError("get_topics() must be implemented.")

    def get_document_topics(self):
        """
        Get topic assignment and distribution for each document.

        Returns:
            dict: Keys are document indices; values are dictionaries with:
                - topic_id (int): Most probable topic
                - topic_score (float): Confidence score of the dominant topic
                - topic_distribution (dict): All topic probabilities for this doc
                Example:
                {
                    0: {
                        "topic_id": 2,
                        "topic_score": 0.91,
                        "topic_distribution": {0: 0.01, 1: 0.03, 2: 0.91, ...}
                    },
                    ...
                }
        """
        raise NotImplementedError("get_document_topics() must be implemented.")

    def get_topic_embeddings(self):
        """
        Return topic-level embeddings, if supported by the model.

        Returns:
            dict or None: A dictionary of topic_id to embedding list, or None if unsupported.
        """
        return None

    def get_topic_scores(self):
        """
        Return importance scores (e.g., frequency or coherence) for each topic.

        Returns:
            dict or None: A dictionary of topic_id to score, or None if unsupported.
        """
        return None

    def get_model_name(self):
        """
        Return the name of the model.

        Returns:
            str: Name of the topic model, used for logging and storage.
        """
        return self.__class__.__name__

    def transform(self, documents, embeddings=None):
        """
        Optional: Assign topics to new unseen documents (if model supports transformation).

        Args:
            documents (List[str]): New documents to transform.
            embeddings (Optional[List[List[float]]]): Embeddings for the new documents (if required).

        Returns:
            dict: Same format as get_document_topics(), mapping new document indices to topic info.
        """
        raise NotImplementedError("transform() is not implemented.")

    @property
    def requires_embeddings(self):
        """
        Whether the model requires embeddings to train or transform.

        Returns:
            bool: True if the model requires embeddings (e.g., BERTopic), False otherwise.
        """
        return False
