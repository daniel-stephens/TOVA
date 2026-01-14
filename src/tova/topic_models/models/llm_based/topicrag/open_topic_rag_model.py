import copy
import json
import logging
import pathlib
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from colorama import Fore, Style  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from tova.prompter.prompter import Prompter
from tova.topic_models.models.llm_based.base import LLMTModel
from tova.utils.cancel import CancellationToken, check_cancel
from tova.utils.progress import ProgressCallback  # type: ignore


class OpenTopicRAGModel(LLMTModel):

    def __init__(
        self,
        model_name: str = None,
        corpus_id: str = None,
        id: str = None,
        model_path: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path(
            "./static/config/config.yaml"),
        load_model: bool = False,
        **kwargs
    ) -> None:

        model_name = model_name if model_name else f"{self.__class__.__name__}_{int(time.time())}"

        super().__init__(model_name, corpus_id, id,
                         model_path, logger, config_path, load_model)

        otr_cfg = self.config.get("opentopicrag", {})
        self.run_from_web = otr_cfg.get("run_from_web", True)
        self.embedding_model_name = otr_cfg.get(
            "embedding_model", "Qwen/Qwen3-Embedding-0.6B")
        self.max_feat_tfidf = otr_cfg.get("max_feat_tfidf", 1000)
        # This value should be a string of preferences provided by the web interface. If running outside the web, it should be retrieved using the get_user_preferences method.
        self.user_preferences = otr_cfg.get("user_preferences", "")
        self.sample = otr_cfg.get("sample", None)
        self.nr_iterations = otr_cfg.get("nr_iterations", 2)

        # prompting setup
        self._tp_generation_prompt_path = otr_cfg.get(
            "tp_generation_prompt_path", "src/tova/topic_models/models/llm_based/topicrag/prompts/tp_generation_dft.txt")
        self._doc_labelling_prompt_path = otr_cfg.get(
            "doc_labelling_prompt_path", "src/tova/topic_models/models/llm_based/topicrag/prompts/doc_labelling_dft.txt")
        self._inference_prompt_path = otr_cfg.get(
            "inference_prompt_path", "src/tova/topic_models/models/llm_based/topicrag/prompts/inference_dft.txt")
        self.num_docs_in_prompt = otr_cfg.get("num_docs_in_prompt", 5)

        with open(self._tp_generation_prompt_path, "r") as file:
            self._tp_generation_prompt = file.read()
        with open(self._doc_labelling_prompt_path, "r") as file:
            self._doc_labelling_prompt = file.read()
        with open(self._inference_prompt_path, "r") as file:
            self._inference_prompt = file.read()

        # Allow overrides
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._prompter = Prompter(
            config_path=self._config_path,
            model_type=self.llm_model_type
        )

        # init embedding model
        self._init_embedding_model()

        if self.run_from_web:
            self._logger.info(
                "Running from web interface. User preferences should be provided via web parameters.")
        else:
            self._logger.info(
                "Running from CLI interface. User preferences will be prompted via stdin.")

    def _init_embedding_model(self) -> None:
        """
        Initializes the embedding model.

        Loads the specified embedding model using Hugging Face transformers. If loading fails, falls back to TF-IDF vectorizer.
        """
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name, trust_remote_code=True)
            self._embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            # Move to GPU if available, otherwise CPU
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self._embedding_model = self._embedding_model.to(self._device)
            self._embedding_model.eval()
            self._logger.info(
                f"Embedding model {self._embedding_model} loaded successfully on {self._device}")
        except Exception as e:
            self._logger.warning(
                f"Error loading embedding model {self._embedding_model}: {e}")
            self._logger.warning("Falling back to TF-IDF for embeddings")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._embedding_model = None
            self._vectorizer = TfidfVectorizer(max_features=1000)

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        """
        if self._embedding_model is not None:
            embeddings = []

            # Process in batches for efficiency with progress bar
            batch_size = 8
            total_batches = (len(texts) + batch_size - 1) // batch_size

            with tqdm(total=total_batches, desc="Creating embeddings", leave=False,
                      bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]

                    # Tokenize and encode
                    with torch.no_grad():
                        encoded_input = self._tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        ).to(self._device)

                        # Get embeddings
                        model_output = self._embedding_model(**encoded_input)

                        # Use mean pooling to get sentence embeddings
                        attention_mask = encoded_input['attention_mask']
                        token_embeddings = model_output.last_hidden_state

                        # Expand attention mask for broadcasting
                        input_mask_expanded = attention_mask.unsqueeze(
                            -1).expand(token_embeddings.size()).float()

                        # Apply mask and mean pooling
                        sum_embeddings = torch.sum(
                            token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(
                            input_mask_expanded.sum(1), min=1e-9)
                        batch_embeddings = (
                            sum_embeddings / sum_mask).cpu().numpy()

                        embeddings.extend(batch_embeddings)

                    pbar.update(1)

            return np.array(embeddings)
        else:
            # Fallback to TF-IDF
            if len(self.document_embeddings) == 0:
                all_texts = [doc['text'] for doc in self.documents]
                self.document_embeddings = self._vectorizer.fit_transform(
                    all_texts)
            return self._vectorizer.transform(texts)

    def _prepare_data(self, model_files: pathlib.Path) -> None:
        """
        Prepares the data by sampling documents (if needed) and creating embeddings. Saves sampled document IDs to a file.
        """

        df_sample = self.df.copy()

        if self.sample:
            if isinstance(self.sample, float):
                df_sample = df_sample.sample(frac=self.sample)
            elif isinstance(self.sample, int):
                df_sample = df_sample.sample(
                    n=min(self.sample, len(df_sample)))
            else:
                raise ValueError(
                    f"Invalid sample type: {type(self.sample)} (expected float or int)")
        self._df_corpus_train = df_sample.copy()

        path_id_samples = model_files / "sampled_doc_ids.txt"
        with path_id_samples.open('w', encoding='utf8') as f:
            for i in df_sample.index:
                f.write(f"{i}\n")

        self._logger.info(
            f"Sampling done. Using {len(df_sample)} docs. Saved idx to {path_id_samples.as_posix()}")

        # Initialize documents topics tracker
        self.documents = [
            {'text': text, 'id': i, 'orig_id': orig_id, 'discovered_topic': None}
            for i, (text, orig_id) in enumerate(zip(df_sample["raw_text"], df_sample["id"]))
        ]

        # Create embeddings for all documents
        self.document_embeddings = self._create_embeddings(
            [doc['text'] for doc in self.documents])

    def _extract_topic_from_response(self, response: str) -> str:
        """Extract the selected topic from LLM response."""
        lines = response.split('\n')
        for line in lines:
            if 'SELECTED TOPIC:' in line:
                return line.split('SELECTED TOPIC:')[1].strip()

        # Fallback: try to find any topic-like phrase
        for line in lines:
            line = line.strip()
            if len(line) > 5 and len(line) < 100 and not line.startswith('Document'):
                return line

        return "general discussion topics"

    def _store_discovered_topics(self, response: str):
        """Store all discovered topics for later analysis."""
        lines = response.split('\n')
        for line in lines:
            if 'DISCOVERED TOPICS:' in line:
                topics_str = line.split('DISCOVERED TOPICS:')[1].strip()
                # Parse the list of topics
                topics = [t.strip() for t in topics_str.split(',')]
                self.discovered_topics.extend(topics)
                break

    def get_user_preferences(self) -> None:
        """
        Get user preferences. This should run differently based on where the model is run from.
        If run from web, preferences are taken from the parameters passed via the web interface.
        If run from CLI, preferences are prompted via stdin.
        """
        if self.run_from_web:
            if not self.user_preferences:
                self._logger.warning(
                    "No user preferences provided via web interface.")
        else:
            print(Fore.GREEN + "\n" + "="*60)
            print("USER PREFERENCE CONFIGURATION")
            print("="*60)
            print("Please specify your preferences for topic generation.")
            print("Examples:")
            print("  - 'Avoid topics about politics and wars'")
            print("  - 'Focus on technology and innovation'")
            print("  - 'No celebrity or entertainment topics'")
            print("  - 'Prefer business and entrepreneurship topics'")
            print("  - Press Enter for no specific preferences")
            print("-"*60 + Style.RESET_ALL)

            prefs = input("Enter your preferences: ").strip()
            self.user_preferences = prefs

        if not self.user_preferences:
            self._logger.warning(
                "Defaulting to empty preferences, all topics will be explored freely.")

    def _generate_and_extract_topic(self) -> Tuple[str, List[Dict]]:
        """
        STEP 1 in the topic generation process: Pick random documents and generate a topic based on user preferences.
        The LLM will discover patterns and extract topics freely.

        Returns
        -------
        Tuple[str, List[Dict]]
            The discovered topic and the list of selected documents used for topic generation.
        """

        with tqdm(total=4, desc="Generating topics") as pbar:

            # Pick random documents
            pbar.set_description("Selecting random documents")
            selected_docs = random.sample(self.documents, min(
                self.num_docs_in_prompt, len(self.documents)))
            time.sleep(0.5)
            pbar.update(1)

            pbar.set_description("Preparing for topic discovery")
            doc_texts = "\n".join([f"Document {i+1}: {doc['text']}"
                                  for i, doc in enumerate(selected_docs)])
            pbar.update(1)

            pbar.set_description("Discovering topics with LLM")

            dicovery_prompt = self._tp_generation_prompt.format(
                doc_texts=doc_texts,
                user_preferences=self.user_preferences,
                self=self
            )

            out, _ = self._prompter.prompt(
                question=dicovery_prompt,
                system_prompt_template_path=None
            )

            topic = self._extract_topic_from_response(out.strip())

            pbar.update(1)

            pbar.set_description("Labeling documents with discovered topic")
            for doc in selected_docs:
                doc['discovered_topic'] = topic
            pbar.update(1)

            pbar.set_description("Topic discovery complete")

        return topic, selected_docs

    def _retrieve_relevant_documents(
        self,
        topic: str,
        selected_docs: List[Dict],
        top_k=10
    ) -> List[Dict]:
        """
        Step 2: RAG retrieval - find documents relevant to the discovered topic.

        Parameters
        ----------
        topic : str
            The discovered topic to retrieve documents for.
        selected_docs : List[Dict]
            The documents originally selected for topic generation.
        top_k : int
            Number of top similar documents to retrieve.

        Returns
        -------
        List[Dict]
            List of retrieved documents with similarity scores.
        """
        with tqdm(total=4, desc="RAG Retrieval", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

            # Create embeddings for the topic
            pbar.set_description(f"Embedding topic: '{topic[:30]}...'")
            topic_embedding = self._create_embeddings([topic])
            pbar.update(1)

            # Calculate similarities
            pbar.set_description("Calculating similarities")
            similarities = cosine_similarity(
                topic_embedding, self.document_embeddings)[0]
            pbar.update(1)

            # Get top-k most similar documents
            pbar.set_description("Selecting top documents")
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            retrieved_docs = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(similarities[idx])
                    retrieved_docs.append(doc)

            # Include originally selected documents
            for doc in selected_docs:
                if doc['id'] not in {d['id'] for d in retrieved_docs}:
                    doc_copy = doc.copy()
                    doc_copy['similarity_score'] = 1.0
                    retrieved_docs.append(doc_copy)

            pbar.update(1)
            pbar.set_description("Retrieval complete")

        self._logger.info(
            f"Retrieved {len(retrieved_docs)} documents for topic: '{topic}'")
        top_scores = [round(d['similarity_score'], 3)
                      for d in retrieved_docs[:3]]
        self._logger.info(f"Top similarity scores: {top_scores}")
        return retrieved_docs

    def _analyze_and_label_documents(
        self,
        topic: str,
        retrieved_docs: List[Dict],
        selected_docs: List[Dict]
    ) -> Dict:
        """
        Step 3: Analyze retrieved documents and assign discovered topics/themes.
        No predefined labels - let LLM discover and assign topics.

        Parameters
        ----------
        topic : str
            The discovered topic.
        retrieved_docs : List[Dict]
            The documents retrieved in the RAG step.
        selected_docs : List[Dict]
            The documents originally selected for topic generation.

        Returns
        -------
        Dict
            Analysis results including primary topic, user preferences, and per-document analysis.
        """

        docs_to_analyze = retrieved_docs[:self.num_docs_in_prompt]

        results = []
        
        def _parse_analysis(response: str) -> Dict:
            """Parse the LLM's analysis response."""
            analysis = {
                'main_topic': 'unclassified',
                'related_themes': [],
                'relevance_score': 0,
                'sentiment': 'neutral',
                'key_entities': []
            }

            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if 'MAIN TOPIC:' in line:
                    analysis['main_topic'] = line.split('MAIN TOPIC:')[
                        1].strip()
                elif 'RELATED THEMES:' in line:
                    themes = line.split('RELATED THEMES:')[1].strip()
                    analysis['related_themes'] = [t.strip()
                                                    for t in themes.split(',')]
                elif 'RELEVANCE SCORE:' in line:
                    try:
                        score = line.split('RELEVANCE SCORE:')[1].strip()
                        analysis['relevance_score'] = float(score)
                    except:
                        analysis['relevance_score'] = 5
                elif 'SENTIMENT:' in line:
                    analysis['sentiment'] = line.split(
                        'SENTIMENT:')[1].strip().lower()
                elif 'KEY ENTITIES:' in line:
                    entities = line.split('KEY ENTITIES:')[1].strip()
                    analysis['key_entities'] = [e.strip()
                                                for e in entities.split(',')]

            return analysis

        with tqdm(total=len(docs_to_analyze), desc="Analyzing documents",
                  bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

            for doc in docs_to_analyze:
                pbar.set_description(
                    f"Analyzing document {pbar.n+1}/{len(docs_to_analyze)}")

                doc_prompt = self._doc_labelling_prompt.format(
                    topic=topic,
                    user_preferences=self.user_preferences,
                    doc_text=doc['text'],
                    self=self
                )

                out, _ = self._prompter.prompt(
                    question=doc_prompt,
                    system_prompt_template_path=None
                )

                analysis = _parse_analysis(out.strip())

                results.append({
                    'id': doc['id'],
                    'text': doc['text'][:200] + "...",
                    'discovered_topic': analysis.get('main_topic', 'unclassified'),
                    'related_themes': analysis.get('related_themes', []),
                    'relevance_score': analysis.get('relevance_score', 0),
                    'sentiment': analysis.get('sentiment', 'neutral'),
                    'key_entities': analysis.get('key_entities', []),
                    'similarity_score': doc.get('similarity_score', 0),
                    'is_selected': doc['id'] in {d['id'] for d in selected_docs}
                })

                pbar.update(1)

            pbar.set_description("Analysis complete")

        return {
            'primary_topic': topic,
            'user_preferences': self.user_preferences,
            'results': results
        }

    def _cleanup_dataset(
        self,
        num_to_remove=10
    ) -> None:
        """
        Step 4: Remove processed documents from the pool.

        Parameters
        ----------
        num_to_remove : int
            Number of documents to remove from the dataset.
        """
        self._logger.info("Starting cleanup process")

        with tqdm(total=1, desc="Cleanup", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
            pbar.set_description("Removing processed documents")

            if len(self.documents) > num_to_remove:
                indices_to_remove = random.sample(
                    range(len(self.documents)), num_to_remove)
                # Sort indices descending to remove without messing up order
                indices_to_remove.sort(reverse=True)

                for idx in indices_to_remove:
                    self.documents.pop(idx)

                # Also remove from embeddings
                if hasattr(self, 'document_embeddings') and len(self.document_embeddings) > 0:
                    try:
                        self.document_embeddings = np.delete(
                            self.document_embeddings, indices_to_remove, axis=0)
                    except Exception as e:
                        self._logger.warning(
                            f"Could not update embeddings during cleanup: {e}")
                        self.document_embeddings = []

                pbar.update(1)
                pbar.set_description("Cleanup complete")
                self._logger.info(
                    f"Removed {num_to_remove} documents. {len(self.documents)} documents remaining.")
            else:
                pbar.update(1)
                self._logger.warning(
                    f"Not enough documents to remove. {len(self.documents)} documents remaining.")

    def _approximate_distributions(
        self,
        all_results: List[Dict],
        all_docs_backup: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Approximates 'traditional' topic model distributions (thetas, betas) and vocabulary. Since during the RAG proccess many documents are never retrieved (and thus never assigned a topic), we implement a simple outlier handling mechanism: we add an extra 'Outlier' topic to which all unseen documents are assigned with probability 1. This ensures that no document has a zero-probability topic distribution, which would break the mathematical assumptions of topic models. The distributions are computed as follows:

        1) thetas. For each document retrieved and labeled by the LLM, we assign a topic distribution based on the 'relevance_score' given by the LLM. 
        If a document was never retrieved (sum of scores = 0), we assign it fully to the 'Outlier' topic.

        2) betas. For each topic, we aggregate all documents assigned to that topic and run TF-IDF to identify the most relevant words for that topic.
        We also add a 'dummy' uniform distribution for the 'Outlier' topic to maintain dimensional consistency.

        3) vocab. The vocabulary is derived from the TF-IDF vectorizer used in step 2. 

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            thetas: Document-topic distribution matrix (D x K+1)
            betas: Topic-word distribution matrix (K+1 x V)
            vocab: List of V words (the vocabulary)
        """

        self._logger.info("Approximating topic model distributions.")

        # unique topics discovered
        unique_topics = sorted(
            list({res['primary_topic'] for res in all_results}))
        topic_to_idx = {t: i for i, t in enumerate(unique_topics)}

        # store topics in the model
        self.topics = {i: t for i, t in enumerate(unique_topics)}
        self.num_topics = len(unique_topics)

        D = len(all_docs_backup)
        doc_id_to_row = {doc['id']: i for i, doc in enumerate(all_docs_backup)}

        #############
        #  thetas   #
        #############
        thetas = np.zeros((D, self.num_topics), dtype=np.float32)
        hits = 0
        for iteration in all_results:
            topic_name = iteration['primary_topic']

            #  skip if topic not recognized
            if topic_name not in topic_to_idx:
                continue

            col_idx = topic_to_idx[topic_name]

            for doc_res in iteration['results']:
                d_id = doc_res.get('id')  # orig id

                # doc must be in our sampled set
                if d_id is not None and d_id in doc_id_to_row:
                    row_idx = doc_id_to_row[d_id]
                    score = doc_res.get('relevance_score', 1.0)

                    # sum since the same doc can appear in different iters
                    thetas[row_idx, col_idx] += score
                    hits += 1

        self._logger.info(
            f"Thetas populated with {hits} document-topic assignments.")

        #############
        # outliers  #
        #############
        thetas_final = np.zeros((D, self.num_topics + 1), dtype=np.float32)

        thetas_final[:, :self.num_topics] = thetas

        row_sums = thetas.sum(axis=1)
        unseen_mask = row_sums < 1e-9  # docs never assigned
        # unassigned docs go fully to outlier topic
        thetas_final[unseen_mask, self.num_topics] = 1.0

        self.topics[self.num_topics] = "Outlier / Unclassified"

        # Normalize rows to sum to 1
        thetas_final = normalize(thetas_final, norm='l1', axis=1)

        num_outliers = np.sum(unseen_mask)
        self._logger.info(
            f"#'classified' docs: {D - num_outliers}. Outliers: {num_outliers}")

        #############
        # betas  #
        #############
        topic_texts = [""] * self.num_topics
        all_texts = [d['text'] for d in all_docs_backup]

        for d in range(D):
            # we only consider the original topics for betas
            active_cols = np.where(thetas_final[d, :self.num_topics] > 0)[0]
            for c in active_cols:
                topic_texts[c] += " " + all_texts[d]

        # tf-idf
        min_df = 2
        max_df = 1.0

        n_topic_docs = len(topic_texts)
        if n_topic_docs <= min_df:
            min_df = 1

        vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )

        try:
            X = vectorizer.fit_transform(topic_texts)
            betas = X.toarray().astype(np.float32)
            vocab = list(vectorizer.get_feature_names_out())
            betas = normalize(betas, norm='l1', axis=1)

            # outlier adjsutment: we add a uniform distribution row for the outlier topic (all words equally likely)
            vocab_size = len(vocab)
            outlier_row = np.full((1, vocab_size), 1.0 /
                                  vocab_size, dtype=np.float32)
            betas_final = np.vstack([betas, outlier_row])

        except ValueError:
            self._logger.warning(
                "TF-IDF vectorization failed during beta approximation. Possibly not enough data per topic.")
            # Fallback: return empty distributions
            return thetas_final, np.zeros((self.num_topics + 1, 1)), ["empty"]

        return thetas_final, betas_final, vocab
    
    def _format_results_by_topic(
        self,
        thetas: np.ndarray,
        all_docs_backup: List[Dict],
        all_results: List[Dict]
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Transforms the raw results into a structured format organized by topics, enriching each document with metadata from the LLM analysis.
        
        Parameters
        ----------
        thetas : np.ndarray
            Document-topic distribution matrix (D x K+1)
        all_docs_backup : List[Dict]
            Original list of all documents used during training.
        all_results : List[Dict]
            Detailed results from each iteration of topic generation and analysis.
        
        Returns
        -------
        List[str], List[str], List[Dict]
            labels: List of topic labels.
            summaries: List of topic summaries. 
            formatted_output: List of topics with their associated documents and enriched metadata.
        """
        
        doc_metadata_map = {}
        for iteration in all_results:
            for doc in iteration['results']:
                meta = {k: v for k, v in doc.items() if k not in ['id', 'relevance_score', 'similarity_score']}
                doc_metadata_map[doc['id']] = meta

        formatted_output = []

        labels = []
        for t_id in sorted(self.topics.keys()):
            topic_name = self.topics[t_id]
            
            topic_entry = {
                "topic_id": t_id,
                "topic_label": topic_name,
                "docs": []
            }

            doc_indices = np.where(thetas[:, t_id] > 0)[0]
            
            temp_docs = []

            for row_idx in doc_indices:
                prob = float(thetas[row_idx, t_id]) # prob only for ordering
                
                original_doc = all_docs_backup[row_idx]
                doc_id = original_doc['id']
                orig_doc_id = original_doc['orig_id']
                
                doc_data = {
                    "doc_id": orig_doc_id,
                }

                # enrich with metadata if available
                if doc_id in doc_metadata_map:
                    doc_data.update(doc_metadata_map[doc_id])
                    # remove "text" if present to avoid redundancy
                    if "text" in doc_data:
                        del doc_data["text"]
                temp_docs.append((prob, doc_data))

            # Sort by prob descending and keep only the dictionary
            temp_docs.sort(key=lambda x: x[0], reverse=True)
            topic_entry["docs"] = [d[1] for d in temp_docs]
            
            formatted_output.append(topic_entry)
            labels.append(topic_name)

        return labels, None, formatted_output

    def train_core(
        self,
        prs: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
        """
        OpenTopicRAG training consists of multiple iterations of topic generation, document retrieval, analysis, and cleanup.
        After all iterations, the model approximates traditional topic model distributions (thetas, betas) and vocabulary.
        
        all_results has the following format: 
        all_results = [
            iteracion_1_dict,
            iteracion_2_dict,
            ...
            iteracion_N_dict
        ]
        where each dict contains:
        * 'primary_topic': str
        * 'user_preferences': str
        * 'results': List of per-document analysis dicts with keys:
            - 'id': original id of the document
            - 'text': excerpt of the document text
            - 'discovered_topic': how the LLM labeled the document (can be different from primary_topic)
            - 'related_themes': List of related themes discovered
            - 'relevance_score': float score assigned by LLM
            - 'sentiment': sentiment assigned by LLM
            - 'key_entities': List of key entities extracted by LLM
            - 'similarity_score': similarity score from RAG retrieval
            - 'is_selected': whether the document was part of the initial selected set for topic generation or was after retrieved via RAG
            
        Parameters
        ----------
        prs : Optional[ProgressCallback], optional
            Progress reporting callback, by default None
        cancel : Optional[CancellationToken], optional
            Cancellation token to allow stopping the process, by default None
        
        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray, List[str], List[str], List[Dict]]
            training_time: Time taken for training in seconds.
            thetas: Document-topic distribution matrix (D x K+1)
            betas: Topic-word distribution matrix (K+1 x V)
            vocab: List of V words (the vocabulary)
            labels: List of topic labels.
            summaries: List of topic summaries. 
            all_results: Detailed results from each iteration.
        """

        if not hasattr(self, "df"):
            raise RuntimeError(
                "Training data not set. Call train_model(data) first.")

        t_start = time.time()

        # 0–10%: Prepare output folder & sample
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.0, 0.1) if prs else None
        prss and prss.report(0.0, "Preparing OpenTopicRAG workspace")

        model_files = self.model_path.joinpath("modelFiles")
        model_files.mkdir(exist_ok=True, parents=True)

        # sampling
        self._prepare_data(model_files)
        # copy is needed since we will modify self.documents during training (cleanup)
        all_docs_backup = copy.deepcopy(self.documents)
        prss and prss.report(1.0, "Preparation completed")
        
        if not self.run_from_web:
            self.get_user_preferences()

        all_results = []
        for _ in tqdm(range(self.nr_iterations), desc="OpenTopicRAG Iterations", bar_format='{l_bar}{bar:30}{r_bar}'):

            # repeat steps 1-4

            # 10–30%: STEP 1: Generate and extract topic
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.1, 0.3) if prs else None
            prss and prss.report(0.0, "Topic generation")
            topic, selected_docs = self._generate_and_extract_topic()
            prss and prss.report(1.0, "Topic generation completed")

            # 30–50%: STEP 2: Retrieve relevant documents
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.3, 0.5) if prs else None
            prss and prss.report(0.0, "Document retrieval")
            retrieved_docs = self._retrieve_relevant_documents(
                topic, selected_docs)
            prss and prss.report(1.0, "Document retrieval completed")

            # 50–70%: STEP 3: Analyze and label documents
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.5, 0.7) if prs else None
            prss and prss.report(0.0, "Labeling documents")
            # This calls the renamed method now
            iteration_results = self._analyze_and_label_documents(
                topic, retrieved_docs, selected_docs)
            all_results.append(iteration_results)
            prss and prss.report(1.0, "Labeling documents completed")

            # 70–90%: STEP 4: CLEANUP
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.7, 0.9) if prs else None
            prss and prss.report(0.0, "Topic correction")
            self._cleanup_dataset()
            prss and prss.report(1.0, "Correction completed")

            # Print results for this iteration
            print("\n Results for this iteration:")
            print(
                f"Primary Topic Discovered: {iteration_results['primary_topic']}")
            print(
                f"User Preferences Applied: {iteration_results['user_preferences']}")
            print("\nDocument Analysis:")
            for j, result in enumerate(iteration_results['results'][:3], 1):
                print(f"\n{j}. Document excerpt: {result['text'][:80]}...")
                print(f"   Discovered Topic: {result['discovered_topic']}")
                print(
                    f"   Related Themes: {', '.join(result['related_themes'][:3])}")
                print(
                    f"   Relevance Score: {result['relevance_score']:.1f}/10")
                print(f"   Sentiment: {result['sentiment']}")
                print(
                    f"   Key Entities: {', '.join(result['key_entities'][:3])}")

        self.all_results = all_results

        # 90–100%: Finalize topics and synthesize distributions
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.9, 1.0) if prs else None
        prss and prss.report(0.9, "Calculating final distributions")

        thetas, betas, vocab = self._approximate_distributions(
            all_results, all_docs_backup)
        labels, summaries, add_info = self._format_results_by_topic(
            thetas=thetas,
            all_docs_backup=all_docs_backup,
            all_results=all_results
        )

        prss and prss.report(1.0, "Training completed")
        
        return time.time() - t_start, thetas, betas, vocab, labels, summaries, add_info

    def infer_core(self, df_infer) -> Tuple[np.ndarray, float]:
        """
        OpenTopicRAG inference consists of performing zero-shot classifying new documents against the learned topics.

        Parameters
        ----------
        df_infer : pd.DataFrame
            DataFrame containing documents to infer topics for. Must have a 'raw_text' column.

        Returns
        -------
        Tuple[np.ndarray, float]
            thetas: Document-topic distribution matrix (D x K+1)
            inference_time: Time taken for inference in seconds.
        """
        if self.topics is None or len(self.topics) == 0:
            raise RuntimeError(
                "Model topics not loaded. Cannot perform inference.")

        t_start = time.time()
        self._logger.info(
            f"Starting inference for {len(df_infer)} documents...")

        K_total = len(self.topics)
        D = len(df_infer)

        # identify the idx of the outlier topic
        outlier_idx = -1
        real_topics_list = []

        for idx, name in self.topics.items():
            if "Outlier" in name and "Unclassified" in name:
                outlier_idx = int(idx)
            else:
                real_topics_list.append(name)

        # inverse map only for real topics
        topic_to_idx = {name: idx for idx,
                        name in self.topics.items() if idx != outlier_idx}

        thetas = np.zeros((D, K_total), dtype=np.float32)
        formatted_topics = "\n".join(
            [f"- {t}" for t in sorted(real_topics_list)])

        doc_texts = df_infer['text'].tolist(
        ) if 'text' in df_infer else df_infer['raw_text'].tolist()

        for idx, text in tqdm(enumerate(doc_texts), total=D, desc="Inferring Topics"):
            snippet = text[:1500]

            try:
                inf_prompt = self._inference_prompt.format(formatted_topics=formatted_topics,doc_text=snippet)

                out, _ = self._prompter.prompt(
                    question=inf_prompt,
                    system_prompt_template_path=None
                )
                predicted_topic = out.strip().strip('".')
                if predicted_topic in topic_to_idx:
                    # llm predics a valid topic
                    col_idx = topic_to_idx[predicted_topic]
                    thetas[idx, col_idx] = 1.0
                
                elif "Outlier" in predicted_topic or "outlier" in predicted_topic.lower():
                    # llm predicts outlier
                    if outlier_idx < K_total:
                        thetas[idx, outlier_idx] = 1.0

                else:
                    # llm predicts hallucination or misspelled topic name. Assign to outlier
                    if outlier_idx < K_total:
                        thetas[idx, outlier_idx] = 1.0

            except Exception as e:
                self._logger.error(
                    f"Error during inference for document {idx}: {e}")
                if outlier_idx < K_total:
                    thetas[idx, outlier_idx] = 1.0

        return thetas, time.time() - t_start

    def save_model(self):
        """
        Save OpenTopicRAG model info to disk.
        """
        model_p = self.model_path
        topics_txt = model_p.joinpath('modelFiles/topics.json')
        topics_add_info = model_p.joinpath('modelFiles/topics_additional_info.json')

        self._logger.info(
            f"Saving topics to {topics_txt.as_posix()}, complete info to {topics_add_info.as_posix()}")

        self._logger.info(f"Saving OpenTopicRAG model to {model_p.as_posix()}")

        if self.topics:
            with topics_txt.open('w', encoding='utf8') as f:
                json.dump(self.topics, f, ensure_ascii=False, indent=2)
    
        if self.all_results:
            with topics_add_info.open('w', encoding='utf8') as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)

        self._logger.info("Model saved successfully!")

    @classmethod
    def from_saved_model(cls, model_path: str):
        """
        Loads a previously saved OpenTopicRAG model from disk.

        Parameters
        ----------
        model_path : str
            Path to the saved model directory.
        Returns
        -------
        cls
            An instance of OpenTopicRAGModel with loaded topics and metadata.
        """

        obj = super().from_saved_model(model_path)

        # load full all_results and topics which are stored separately
        topics_add_info = pathlib.Path(
            model_path) / 'modelFiles/topics_additional_info.json'
        if topics_add_info.exists():
            try:
                with open(topics_add_info, 'r', encoding='utf-8') as f:
                    obj.all_results = json.load(f)
                obj._logger.info("Detailed results loaded from json.")
            except Exception as e:
                obj._logger.warning(f"Could not load detailed results: {e}")
        else:
            obj.all_results = []

        topics_info = pathlib.Path(model_path) / 'modelFiles/topics.json'
        if topics_info.exists():
            try:
                with open(topics_info, 'r', encoding='utf-8') as f:
                    obj.topics = json.load(f)
                obj.num_topics = len(obj.topics)
                obj._logger.info("Topics loaded from json.")
            except Exception as e:
                obj._logger.warning(f"Could not load topics: {e}")
        else:
            obj.topics = {}
        obj.num_topics = len(obj.topics)

        return obj

    def print_topics(self, verbose: bool = False) -> list:
        """
        Print the list of topics for the topic model.

        Parameters
        ----------
        verbose : bool, optional
            If True, print the topics to the console, by default False.

        Returns
        -------
        list
            List with the keywords for each topic.
        """

        if self.topics is None:
            self._logger.warning("Topics not loaded yet.")
            return {}

        if verbose:
            for k, v in self.topics.items():
                print(f"Topic {k}: {v}")
        return self.topics or {}
