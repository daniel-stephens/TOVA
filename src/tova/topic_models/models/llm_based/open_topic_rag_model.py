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
from typing import List, Optional, Tuple
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
        if self.run_from_web:
            self.logger.info(
                "Running from web interface. User preferences should be provided via web parameters.")
        else:
            self.logger.info(
                "Running from CLI interface. User preferences will be prompted via stdin.")
        self.embedding_model = otr_cfg.get(
            "embedding_model", "Qwen/Qwen3-Embedding-0.6B")
        self.max_feat_tfidf = otr_cfg.get("max_feat_tfidf", 1000)
        # This value should be a string of preferences provided by the web interface. If running outside the web, it should be retrieved using the get_user_preferences method.
        self.user_preferences = otr_cfg.get("user_preferences", "")
        self.sample = otr_cfg.get("sample", None)

        # prompting setup
        tp_generation_prompt_path = otr_cfg.get(
            "otrag_prompt_path", "src/tova/prompter/prompts/otrag_dft.txt")
        with open(tp_generation_prompt_path, "r") as file:
            self.tp_generation_prompt = file.read()
        doc_labelling_prompt_path = otr_cfg.get(
            "doc_labelling_prompt_path", "src/tova/prompter/prompts/otrag_doc_labelling_dft.txt")
        with open(doc_labelling_prompt_path, "r") as file:
            self.doc_labelling_prompt = file.read()
        self.prompter = Prompter(
            config_path=self._config_path,
            model_type=self.llm_model_type
        )
        self.num_docs_in_prompt = otr_cfg.get("num_docs_in_prompt", 5)

        # Allow overrides
        for k, v in kwargs.items():
            setattr(self, k, v)

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

    def get_user_preferences(self) -> None:
        """
        Get user preferences. This should run differently based on where the model is run from.
        If run from web, preferences are taken from the parameters passed via the web interface.
        If run from CLI, preferences are prompted via stdin.
        """
        if self.run_from_web:
            if not self.user_preferences:
                self.logger.warning(
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
            self.logger.warning(
                "Defaulting to empty preferences, all topics will be explored freely.")

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
            doc_texts = "\n".join([f"Document {i+1}: {doc['raw_text']}"
                                  for i, doc in enumerate(selected_docs)])
            pbar.update(1)

            pbar.set_description("Discovering topics with LLM")

            dicovery_prompt = self.prompt.format(
                doc_texts=doc_texts,
                user_preferences=self.user_preferences,
            )

            out, _ = self.prompter.prompt(
                question=dicovery_prompt,
                system_prompt_template_path=None
            )
            
            def _extract_topic_from_response(response: str) -> str:
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

            topic = _extract_topic_from_response(out.strip())

            pbar.update(1)

            pbar.set_description("Labeling documents with discovered topic")
            for doc in selected_docs:
                doc['discovered_topic'] = topic
            pbar.update(1)

            pbar.set_description("Topic discovery complete")
