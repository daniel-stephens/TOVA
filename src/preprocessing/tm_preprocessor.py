# @TODO: @lcalvobartolome

# def get_bow(self, vocab: List[str]) -> np.ndarray:
#     """
#     Get the Bag of Words (BoW) matrix of the documents, maintaining the internal order of the words as in the betas matrix.

#     Parameters
#     ----------
#     vocab : List[str]
#         The vocabulary of the model.

#     Returns
#     -------
#     np.ndarray
#         The Bag of Words matrix.
#     """

#     if self.train_data is None:
#         self._logger.error("Train data not loaded. Cannot create BoW matrix.")
#         return np.array([])

#     # Build vocab mappings
#     vocab_w2id = {wd: id_wd for id_wd, wd in enumerate(vocab)}
#     vocab_id2w = {str(id_wd): wd for id_wd, wd in enumerate(vocab)}

#     # Create gensim dictionary and BoW representation
#     gensim_dict = Dictionary(self.train_data)
#     bow = [gensim_dict.doc2bow(doc) for doc in self.train_data]

#     # Map gensim word ids to vocabulary ids (filtering unknowns)
#     gensim_to_tmt_ids = {
#         word_id: vocab_w2id[gensim_dict[word_id]]
#         for doc in bow for word_id, _ in doc
#         if gensim_dict[word_id] in vocab_w2id
#     }

#     # Sort BoW entries by vocab id
#     sorted_bow = [
#         sorted(
#             [
#                 (gensim_to_tmt_ids[gensim_word_id], weight)
#                 for gensim_word_id, weight in doc
#                 if gensim_word_id in gensim_to_tmt_ids
#             ],
#             key=lambda x: x[0]
#         )
#         for doc in bow
#     ]

#     # Fill the BoW matrix
#     bow_mat = np.zeros((len(sorted_bow), len(vocab)), dtype=np.int32)
#     for doc_id, doc in enumerate(sorted_bow):
#         for word_id, weight in doc:
#             np.put(bow_mat[doc_id], word_id, weight)

#     self._logger.info(f"BoW matrix shape: {bow_mat.shape}")

#     return bow_mat