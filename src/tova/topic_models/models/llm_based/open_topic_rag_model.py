import json
import logging
import os
import pathlib
import random
import re
import time
import numpy as np
import torch
from collections import defaultdict
from subprocess import CalledProcessError, check_output
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
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
        if self.run_from_web:
            self._logger.info(
                "Running from web interface. User preferences should be provided via web parameters.")
        else:
            self._logger.info(
                "Running from CLI interface. User preferences will be prompted via stdin.")
        self.embedding_model = otr_cfg.get(
            "embedding_model", "Qwen/Qwen3-Embedding-0.6B")
        self.max_feat_tfidf = otr_cfg.get("max_feat_tfidf", 1000)
        # This value should be a string of preferences provided by the web interface. If running outside the web, it should be retrieved using the get_user_preferences method.
        self.user_preferences = otr_cfg.get("user_preferences", "")
        self.sample = otr_cfg.get("sample", None)
        self.nr_iterations = otr_cfg.get("nr_iterations", 2)

        # prompting setup
        tp_generation_prompt_path = otr_cfg.get(
            "tp_generation_prompt_path", "src/tova/prompter/prompts/otrag_dft.txt")
        doc_labelling_prompt_path = otr_cfg.get(
            "doc_labelling_prompt_path", "src/tova/prompter/prompts/otrag_doc_labelling_dft.txt")
        self.num_docs_in_prompt = otr_cfg.get("num_docs_in_prompt", 5)
        
        with open(tp_generation_prompt_path, "r") as file:
            self.tp_generation_prompt = file.read()
        with open(doc_labelling_prompt_path, "r") as file:
            self.doc_labelling_prompt = file.read()
            
        # Allow overrides
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.prompter = Prompter(
            config_path=self._config_path,
            model_type=self.llm_model_type
        )
        
        # init embedding model
        self._init_embedding_model()

    def _init_embedding_model(self) -> None:
        """
        Initialize the embedding model.

        Loads the specified embedding model using Hugging Face transformers. If loading fails, falls back to TF-IDF vectorizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model, trust_remote_code=True)
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            # Move to GPU if available, otherwise CPU
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_model = self.embedding_model.to(self.device)
            self.embedding_model.eval()
            self._logger.info(
                f"Embedding model {self.embedding_model} loaded successfully on {self.device}")
        except Exception as e:
            self._logger.warning(
                f"Error loading embedding model {self.embedding_model}: {e}")
            self._logger.warning("Falling back to TF-IDF for embeddings")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.embedding_model = None
            self.vectorizer = TfidfVectorizer(max_features=1000)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        """
        if self.embedding_model is not None:
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
                        encoded_input = self.tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        ).to(self.device)

                        # Get embeddings
                        model_output = self.embedding_model(**encoded_input)

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
                self.document_embeddings = self.vectorizer.fit_transform(
                    all_texts)
            return self.vectorizer.transform(texts)

    def _prepare_data(self, model_files: pathlib.Path) -> pathlib.Path:
        """
        Prepare data for OpenTopicRAG by sampling (if needed) and saving to JSONL.
        Returns the path to the prepared data file.
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
        # TODO: it may be better to just save the indices of the sampled docs
        path_sample = model_files / \
            f"sample_{str(self.sample)}.jsonl" if self.sample else model_files / \
            "sample.jsonl"
        df_sample.to_json(path_sample, lines=True, orient="records")
        self._logger.info(
            f"Sampling done. Using {len(df_sample)} docs. Saved to {path_sample.as_posix()}")

        #  initialize documents topics tracker
        self.documents = [
            {'text': text, 'id': i, 'discovered_topic': None}
            for i, text in enumerate(df_sample["raw_text"])
        ]
        
        # create embeddings for all documents
        self.document_embeddings = self.create_embeddings(
            [doc['text'] for doc in self.documents])
        
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
            print("\n" + "="*60)
            print("USER PREFERENCE CONFIGURATION")
            print("="*60)
            print("Please specify your preferences for topic generation.")
            print("Examples:")
            print("  - 'Avoid topics about politics and wars'")
            print("  - 'Focus on technology and innovation'")
            print("  - 'No celebrity or entertainment topics'")
            print("  - 'Prefer business and entrepreneurship topics'")
            print("  - Press Enter for no specific preferences")
            print("-"*60)

            prefs = input("Enter your preferences: ").strip()
            self.user_preferences = prefs

        if not self.user_preferences:
            self._logger.warning(
                "Defaulting to empty preferences, all topics will be explored freely.")

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
            # Implementation for storing topics
            pass

    def generate_and_extract_topic(self, iter=4):

        with tqdm(range(iter), desc="Generating topics") as pbar:

            # Pick random documents
            pbar.set_description("Selecting random documents")
            selected_docs = random.sample(self.documents, min(
                self.num_docs_in_prompt, len(self.documents)))
            time.sleep(0.5)
            pbar.update(1)

            # Create prompt for open topic discovery
            pbar.set_description("Preparing for topic discovery")
            doc_texts = "\n".join([f"Document {i+1}: {doc['text']}"
                                  for i, doc in enumerate(selected_docs)])
            pbar.update(1)

            pbar.set_description("Discovering topics with LLM")

            dicovery_prompt = self.tp_generation_prompt.format(
                doc_texts=doc_texts,
                user_preferences=self.user_preferences,
                self=self 
            )

            out, _ = self.prompter.prompt(
                question=dicovery_prompt,
                system_prompt_template_path=None
            )

            topic = self._extract_topic_from_response(out.strip())

            self._store_discovered_topics(out.strip())

            pbar.update(1)

            pbar.set_description("Labeling documents with discovered topic")
            for doc in selected_docs:
                doc['discovered_topic'] = topic
            pbar.update(1)

            pbar.set_description("Topic discovery complete")

        return topic, selected_docs

    def retrieve_relevant_documents(self, topic: str, selected_docs: List[Dict], top_k=10) -> List[Dict]:

        with tqdm(total=4, desc="RAG Retrieval", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

            # Create embeddings for the topic
            pbar.set_description(f"Embedding topic: '{topic[:30]}...'")
            topic_embedding = self.create_embeddings([topic])
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
            selected_ids = {doc['id'] for doc in selected_docs}
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
    
    def analyze_and_label_documents(self, topic: str, retrieved_docs: List[Dict], selected_docs: List[Dict]) -> Dict:
        
        docs_to_analyze = retrieved_docs[:self.num_docs_in_prompt]
        
        results = []

        with tqdm(total=len(docs_to_analyze), desc="Analyzing documents", 
            bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
    
            for doc in docs_to_analyze:
                pbar.set_description(f"Analyzing document {pbar.n+1}/{len(docs_to_analyze)}")
                
                doc_prompt = self.doc_labelling_prompt.format(
                    topic=topic,
                    user_preferences=self.user_preferences,
                    doc_text=doc['text'],
                    self=self
                )
                
                out, _ = self.prompter.prompt(
                    question=doc_prompt,
                    system_prompt_template_path=None
                )
                
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
                        analysis['main_topic'] = line.split('MAIN TOPIC:')[1].strip()
                    elif 'RELATED THEMES:' in line:
                        themes = line.split('RELATED THEMES:')[1].strip()
                        analysis['related_themes'] = [t.strip() for t in themes.split(',')]
                    elif 'RELEVANCE SCORE:' in line:
                        try:
                            score = line.split('RELEVANCE SCORE:')[1].strip()
                            analysis['relevance_score'] = float(score)
                        except:
                            analysis['relevance_score'] = 5
                    elif 'SENTIMENT:' in line:
                        analysis['sentiment'] = line.split('SENTIMENT:')[1].strip().lower()
                    elif 'KEY ENTITIES:' in line:
                        entities = line.split('KEY ENTITIES:')[1].strip()
                        analysis['key_entities'] = [e.strip() for e in entities.split(',')]
                
                return analysis
            
            analysis = _parse_analysis(out.strip())
            
            results.append({
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
        
    def cleanup_dataset(self, num_to_remove=10):
        """
        Step 4: Remove processed documents from the pool.
        """
        self._logger.info("Starting cleanup process")

        with tqdm(total=1, desc="Cleanup", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
            pbar.set_description("Removing processed documents")

            if len(self.documents) > num_to_remove:
                indices_to_remove = random.sample(range(len(self.documents)), num_to_remove)
                # Sort indices descending to remove without messing up order
                indices_to_remove.sort(reverse=True)

                for idx in indices_to_remove:
                    self.documents.pop(idx)

                # FIX: Actualizar self.document_embeddings en lugar de borrarlo
                # Si se borra ([]), la siguiente iteración falla en retrieve_relevant_documents
                if hasattr(self, 'document_embeddings') and len(self.document_embeddings) > 0:
                    try:
                        # Asumiendo que es un numpy array basado en create_embeddings
                        self.document_embeddings = np.delete(self.document_embeddings, indices_to_remove, axis=0)
                    except Exception as e:
                        self._logger.warning(f"Could not update embeddings during cleanup: {e}")
                        self.document_embeddings = [] # Fallback original logic if delete fails

                pbar.update(1)
                pbar.set_description("Cleanup complete")
                self._logger.info(f"Removed {num_to_remove} documents. {len(self.documents)} documents remaining.")
            else:
                pbar.update(1)
                self._logger.warning(f"Not enough documents to remove. {len(self.documents)} documents remaining.")
        
    def train_core(
        self,
        prs: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:

        if not hasattr(self, "df"):
            raise RuntimeError(
                "Training data not set. Call train_model(data) first.")

        t_start = time.time()

        # 0–10%: Prepare output folder & sample
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.0, 0.1) if prs else None
        prss and prss.report(0.0, "Preparing TopicGPT workspace")

        model_files = self.model_path.joinpath("modelFiles")
        model_files.mkdir(exist_ok=True, parents=True)

        # sampling
        self._prepare_data(model_files)

        prss and prss.report(1.0, "Preparation completed")
        
        all_results = []
        for _ in tqdm(range(self.nr_iterations), desc="OpenTopicRAG Iterations", bar_format='{l_bar}{bar:30}{r_bar}'):
            
            # repeat steps 1-4

            # 10–30%: STEP 1: Generate and extract topic
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.1, 0.3) if prs else None
            prss and prss.report(0.0, "Topic generation")
            topic, selected_docs = self.generate_and_extract_topic()
            prss and prss.report(1.0, "Topic generation completed")

            # 30–50%: STEP 2: Retrieve relevant documents
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.3, 0.5) if prs else None
            prss and prss.report(0.0, "Document retrieval")
            retrieved_docs = self.retrieve_relevant_documents(topic, selected_docs)
            prss and prss.report(1.0, "Document retrieval completed")

            # 50–70%: STEP 3: Analyze and label documents
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.5, 0.7) if prs else None
            prss and prss.report(0.0, "Labeling documents")
            # This calls the renamed method now
            iteration_results = self.analyze_and_label_documents(topic, retrieved_docs, selected_docs)
            all_results.append(iteration_results)
            prss and prss.report(1.0, "Labeling documents completed")

            # 70–90%: STEP 4: CLEANUP
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.7, 0.9) if prs else None
            prss and prss.report(0.0, "Topic correction")
            self.cleanup_dataset()
            prss and prss.report(1.0, "Correction completed")
            
            # Print results for this iteration
            print("\n📊 Results for this iteration:")
            print(f"Primary Topic Discovered: {iteration_results['primary_topic']}")
            print(f"User Preferences Applied: {iteration_results['user_preferences']}")
            print("\nDocument Analysis:")
            for j, result in enumerate(iteration_results['results'][:3], 1):
                print(f"\n{j}. Document excerpt: {result['text'][:80]}...")
                print(f"   Discovered Topic: {result['discovered_topic']}")
                print(f"   Related Themes: {', '.join(result['related_themes'][:3])}")
                print(f"   Relevance Score: {result['relevance_score']:.1f}/10")
                print(f"   Sentiment: {result['sentiment']}")
                print(f"   Key Entities: {', '.join(result['key_entities'][:3])}")

        # 90–100%: Finalize topics and synthesize distributions
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.9, 1.0) if prs else None
        prss and prss.report(0.9, "Finalizing topics & synthetic distributions")

        import pdb; pdb.set_trace()
        #topics = self._read_topics(outputs["generation_topic"])
        #self.topics = topics
        #print(f"Extracted {len(topics)} topics.")

        # Save the raw topic strings
        #with self.model_path.joinpath('orig_tpc_descriptions.txt').open('w', encoding='utf8') as fout:
       #     fout.write('\n'.join([topics[k] for k in sorted(topics.keys())]))

        #thetas, betas, vocab = self._approximate_distributions()
        thetas, betas, vocab = np.random.rand(10, 5), np.random.rand(10, 1000), [f"word{i}" for i in range(1000)]
        prss and prss.report(1.0, "Topics & synthetic distributions ready")

        return time.time() - t_start, thetas, betas, vocab
    
    # FIX: Se añadió self
    def infer_core(self):
        print("infer")
    
    @classmethod
    def from_saved_model(cls, model_path: str):
        print("saved")
        
    def save_model(self) -> None:
        print("save")
    
    def print_topics(self, verbose: bool = False, get_second_level: bool = False) -> dict:
        print("print")