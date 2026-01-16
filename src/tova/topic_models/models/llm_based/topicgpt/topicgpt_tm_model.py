import json
import logging
import os
import pathlib
import re
import time
from collections import defaultdict
from subprocess import CalledProcessError, check_output
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import topicgpt_python as tg  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore

from tova.topic_models.models.llm_based.base import LLMTModel
from tova.utils.cancel import CancellationToken, check_cancel
from tova.utils.progress import ProgressCallback  # type: ignore

_TOPICS_LINE_RE = re.compile(r"^\s*\[\d+\]\s*([^(]+?)(?:\s*\(|\s*:|$)")
_RESP_RE = re.compile(r"^\s*\[(\d+)\]\s*([^:]+)\s*:")


class TopicGPTTMmodel(LLMTModel):
    """
    Wraps the TopicGPT script pipeline (generation/refinement/
    assignment/correction [+ optional 2nd-level]) within the same training/
    inference structure as TomotopyLDATMmodel.
    """

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

        tg_cfg = self.config.get("topicgpt", {})
        self.sample = tg_cfg.get("sample", 0.001)
        self.temperature = float(tg_cfg.get("temperature", 0.0))
        self.top_p = float(tg_cfg.get("top_p", 0.0))
        self.max_tokens_gen1 = int(tg_cfg.get("max_tokens_gen1", 300))
        self.max_tokens_gen2 = int(tg_cfg.get("max_tokens_gen2", 500))
        self.max_tokens_assign = int(tg_cfg.get("max_tokens_assign", 300))
        self.refined_again = bool(tg_cfg.get("refined_again", False))
        self.remove = bool(tg_cfg.get("remove", False))
        self.do_second_level = bool(tg_cfg.get("do_second_level", False))
        self.verbose_scripts = bool(tg_cfg.get("verbose", True))

        # Allow overrides
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Paths to prompts
        cwd = pathlib.Path(os.getcwd())
        self._p_prompts = cwd / "src/tova/topic_models/models/llm_based/topicGPT/prompt"

        self._generation_prompt = self._p_prompts / "generation_1.txt"
        self._seed_1 = self._p_prompts / "seed_1.md"
        self._refinement_prompt = self._p_prompts / "refinement.txt"
        self._generation_2_prompt = self._p_prompts / "generation_2.txt"
        self._assignment_prompt = self._p_prompts / "assignment.txt"
        self._correction_prompt = self._p_prompts / "correction.txt"

        # Will be resolved to files under model_path/modelFiles at train time
        self._outputs_save = {
            "generation_out": "generation_1.jsonl",
            "generation_topic": "generation_1.md",
            "refinement_out": "refinement.jsonl",
            "refinement_topic": "refinement.md",
            "refinement_mapping": "refinement_mapping.txt",
            "refinement_updated": "refinement_updated.jsonl",
            "generation_2_out": "generation_2.jsonl",
            "generation_2_topic": "generation_2.md",
            "assignment_out": "assignment.jsonl",
            "correction_out": "assignment_corrected.jsonl",
        }

        self.topics: Optional[dict] = None
        self.second_topics: Optional[dict] = None

        self._logger.info(
            f"{self.__class__.__name__} initialized, "
            f"sample={self.sample}, llm_provider='{self.llm_provider}', llm_model_type='{self.llm_model_type}', "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"do_second_level={self.do_second_level}."
        )

    def _prepare_data(self, model_files: pathlib.Path) -> pathlib.Path:
        """
        Prepare data files for TopicGPT scripts: full.jsonl (all docs)  and
        sample_*.jsonl (sampled docs). Returns paths to (sampled, full).
        """

        df_full = self.df.copy().rename(columns={"raw_text": "text"})
        path_full = model_files / "full.jsonl"
        df_full.to_json(path_full, lines=True, orient="records")

        df_sample = self.df.copy()
        # gpt expects 'text' field
        df_sample = df_sample.rename(columns={"raw_text": "text"})
        if self.sample:
            if isinstance(self.sample, float):
                df_sample = df_sample.sample(frac=self.sample)
            elif isinstance(self.sample, int):
                df_sample = df_sample.sample(
                    n=min(self.sample, len(df_sample)))
            else:
                raise ValueError(
                    f"Invalid sample type: {type(self.sample)} (expected float or int)")
        path_sample = model_files / \
            f"sample_{str(self.sample)}.jsonl" if self.sample else model_files / \
            "sample.jsonl"
        df_sample.to_json(path_sample, lines=True, orient="records")
        self._logger.info(
            f"Sampling done. Using {len(df_sample)} docs. Saved to {path_sample.as_posix()}")

        return path_sample, path_full

    def _topic_label_from_topic_line(self, line: str) -> str:
        # "[1] Healthcare (Count: 1): ..." -> "Healthcare"
        m = _TOPICS_LINE_RE.match(line or "")
        return self._norm(m.group(1)) if m else self._norm(line)

    def _topic_label_from_response(self, resp: str) -> str:
        # "[1] Immigration: ..." -> ("1", "Immigration")
        m = _RESP_RE.match(resp or "")
        return self._norm(m.group(2)) if m else ""

    def _norm(self, s: str) -> str:
        return " ".join((s or "").strip().lower().split())

    def _run_cmd(self, cmd: str, banner: str):
        """
        Run a shell command with logging; raise on failure.
        """
        self._logger.info(banner)
        self._logger.info(f"-- -- Running command: {cmd}")
        try:
            check_output(args=cmd, shell=True)
        except CalledProcessError as e:
            self._logger.error(f"Command failed (returncode {e.returncode}).")
            self._logger.error(e.output.decode("utf-8")
                               if e.output else "<no output>")
            raise
        except Exception as e:
            self._logger.exception("Failed to run external script.")
            raise

    def _read_topics(self, path: pathlib.Path) -> dict:
        """
        Read topic lines (non-empty) from a .md file into {idx: line}.
        """
        with open(path, "r", encoding="utf8") as fin:
            lines = [ln.strip() for ln in fin.readlines() if ln.strip()]
        topics = {i: ln for i, ln in enumerate(lines)}
        self._logger.info(f"Loaded {len(topics)} topics from {path.name}")
        return topics

    def _approximate_thetas(self, assign_path: pathlib.Path, df: pd.DataFrame) -> np.ndarray:
        """Approximates document-topic distribution matrix (thetas) from
        assignment_corrected.jsonl file.

        Parameters
        ----------
        assign_path : Path
            Path to assignment_corrected.jsonl file.

        Returns
        -------
        thetas : np.ndarray
            Document-topic distribution matrix (D x T).
        """

        if self.topics is None:
            raise RuntimeError("Topics not available. Run training first.")

        K = len(self.topics)
        # map normalized topic name -> index (level 1)
        topic_name_to_idx = {
            self._topic_label_from_topic_line(name): k
            for k, name in self.topics.items()
        }
        self._logger.info(f"Topic name to index mapping: {topic_name_to_idx}")

        doc_ids = df.id.values.tolist()

        D = len(doc_ids)
        df_ids = set(doc_ids)
        thetas = np.zeros((D, K), dtype=np.float32)

        # construct thetas from assignments
        with open(assign_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = obj["id"]
                if doc_id not in df_ids:
                    continue

                resp = obj.get("responses", "")
                if not isinstance(resp, str):
                    continue

                m = _RESP_RE.match(resp)
                if not m:
                    continue

                level = int(m.group(1))
                if level != 1:
                    continue

                topic_name = self._topic_label_from_response(resp)
                k = topic_name_to_idx.get(topic_name)
                if k is not None:
                    d = doc_ids.index(doc_id)
                    thetas[d, k] = 1.0

        # normalize rows
        row_sums = thetas.sum(axis=1, keepdims=True)
        nz = row_sums.squeeze() > 0
        thetas[nz] /= row_sums[nz]

        self._logger.info(
            f"Synthetic thetas built: theta={thetas.shape}"
        )

        return thetas

    def _approximate_betas(self, thetas: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Approximates topic-word distribution matrix (betas) and vocabulary
        from TF-IDF on 'pseudo-documents' formed by concatenating texts of
        documents assigned to each topic.

        Parameters
        ----------
        thetas : np.ndarray
            Document-topic distribution matrix (D x T).

        Returns
        -------
        betas : np.ndarray
            Topic-word distribution matrix (T x V).
        vocab : List[str]
            Vocabulary list.
        """

        K = thetas.shape[1]
        doc_texts = df.raw_text.values.tolist()

        # pseudo-betas via TF-IDF on topic documents
        topic_docs = defaultdict(list)
        for d, text in enumerate(doc_texts):
            ks = np.where(thetas[d] > 0)[0]
            for k in ks:
                topic_docs[k].append(text)

        topic_texts = [
            " ".join(topic_docs[k]) if k in topic_docs else ""
            for k in range(K)
        ]

        n_topic_docs = len(topic_texts)

        min_df = 2
        max_df = 1.0

        if n_topic_docs <= min_df:
            min_df = 1

        vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )

        X = vectorizer.fit_transform(topic_texts)
        betas = X.toarray().astype(np.float32)
        vocab = list(vectorizer.get_feature_names_out())
        betas = normalize(betas, norm='l1', axis=1)

        self._logger.info(
            f"Synthetic betas built: beta={betas.shape}"
        )

        return betas, vocab

    def _approximate_distributions(self, assign_path: pathlib.Path, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Approximates 'traditional' topic model distributions (thetas, betas) and vocabulary as follows:

        1) thetas (from assignment_corrected.jsonl after correction step). After parsing [1] topics, each document d has a set of assigned topics. We then define theta[d, k] = 1/|T_d| if k in T_d, else 0, and row-normalize.
        2) betas. We treat each topic as a 'pseudo-document' formed by concatenating the texts of all documents assigned to that topic. We then fit a TF-IDF vectorizer on these pseudo-documents to get a topic-term matrix, which we L1-normalize row-wise to get betas.
        3) vocab is simply the feature list learned by the TF-IDF vectorizer, aligned with the columns of betas.

        Parameters
        ----------
        assign_path : Path
            Path to assignment_corrected.jsonl file.
        df : pd.DataFrame
            The training dataframe with 'raw_text' and 'id' columns.
        
        Returns
        -------
        thetas : np.ndarray
            Document-topic distribution matrix (D x T).
        betas : np.ndarray
            Topic-word distribution matrix (T x V).
        vocab : List[str]
            Vocabulary list.
        """

        if self.topics is None:
            raise RuntimeError("Topics not available. Run training first.")

        K = len(self.topics)
        # map normalized topic name -> index (level 1)
        topic_name_to_idx = {
            self._topic_label_from_topic_line(name): k
            for k, name in self.topics.items()
        }
        self._logger.info(f"Topic name to index mapping: {topic_name_to_idx}")

        thetas = self._approximate_thetas(assign_path, df)

        # pseudo-betas via TF-IDF on topic documents
        betas, vocab = self._approximate_betas(thetas, df)

        self._logger.info(
            f"Synthetic distributions built: theta={thetas.shape}, beta={betas.shape}"
        )

        return thetas, betas, vocab
    
    def _format_results_by_topic(self) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Formats topic labels, summaries, and additional info from the
        trained TopicGPT model.

        Returns
        -------
        labels : List[str]
            List of topic labels.
        summaries : List[str]
            List of topic summaries.
        add_info : List[Dict]
            List of additional information per topic.
        """
        if not hasattr(self, "topics") or self.topics is None:
            raise RuntimeError("Topics not available. Run training first.")

        labels = []
        summaries = []
        add_info = None # Additional info is None for TopicGPT; placeholder for compatibility

        for k in sorted(self.topics.keys()):
            topic_line = self.topics[k]
            label = self._topic_label_from_topic_line(topic_line).capitalize()
            labels.append(label)

            # the summary is the full topic line after the ":"
            summary = topic_line.split(":")[-1].strip()
            summaries.append(summary)
            
        return labels, summaries, add_info

    def train_core(
        self,
        prs: Optional[ProgressCallback] = None,
        cancel: Optional[CancellationToken] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
        """
        Execute TopicGPT script pipeline and return synthetic (thetas, betas, vocab)
        so the base class can continue. Also writes the raw TopicGPT artifacts
        to disk and records topic strings for `print_topics`.
        """
        if not hasattr(self, "df"):
            raise RuntimeError(
                "Training data not set. Call train_model(data) first.")

        t_start = time.time()

        # 0–5%: Prepare output folder & sample
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.0, 0.05) if prs else None
        prss and prss.report(0.0, "Preparing TopicGPT workspace")

        model_files = self.model_path.joinpath("modelFiles")
        model_files.mkdir(exist_ok=True, parents=True)

        # resolve outputs to this folder
        outputs = {k: model_files / v for k, v in self._outputs_save.items()}

        # sampling
        path_sample, path_full = self._prepare_data(model_files)

        prss and prss.report(1.0, "Preparation completed")

        # 5–25%: TOPIC GENERATION
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.05, 0.25) if prs else None
        prss and prss.report(0.0, "Topic generation")
        tg.generate_topic_lvl1(
            api=self.llm_provider,
            model=self.llm_model_type,
            data=path_sample.as_posix(),
            prompt_file=self._generation_prompt.as_posix(),
            seed_file=self._seed_1.as_posix(),
            out_file=outputs['generation_out'].as_posix(),
            topic_file=outputs['generation_topic'].as_posix(),
            verbose=self.verbose_scripts
        )
        prss and prss.report(1.0, "Generation completed")

        # 25–45%: TOPIC REFINEMENT
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.25, 0.45) if prs else None
        prss and prss.report(0.0, "Topic refinement")
        tg.refine_topics(
            api=self.llm_provider,
            model=self.llm_model_type,
            prompt_file=self._refinement_prompt.as_posix(),
            generation_file=outputs['generation_out'].as_posix(),
            topic_file=outputs['generation_topic'].as_posix(),
            out_file=outputs['refinement_topic'].as_posix(),
            updated_file=outputs['refinement_out'].as_posix(),
            verbose=self.verbose_scripts,
            remove=self.remove,
            mapping_file=outputs['refinement_mapping'].as_posix(),
        )
        prss and prss.report(1.0, "Refinement completed")

        # 45–65%: TOPIC ASSIGNMENT
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.45, 0.65) if prs else None
        prss and prss.report(0.0, "Topic assignment")
        tg.assign_topics(
            api=self.llm_provider,
            model=self.llm_model_type,
            data=path_full.as_posix(),  # assign on full data
            prompt_file=self._assignment_prompt.as_posix(),
            topic_file=outputs['generation_topic'].as_posix(),
            out_file=outputs['assignment_out'].as_posix(),
            verbose=self.verbose_scripts
        )
        prss and prss.report(1.0, "Assignment completed")

        # 65–80%: TOPIC CORRECTION
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.65, 0.8) if prs else None
        prss and prss.report(0.0, "Topic correction")
        tg.correct_topics(
            api=self.llm_provider,
            model=self.llm_model_type,
            data_path=outputs['assignment_out'].as_posix(),
            prompt_path=self._correction_prompt.as_posix(),
            topic_path=outputs['generation_topic'].as_posix(),
            output_path=outputs['correction_out'].as_posix(),
            verbose=self.verbose_scripts
        )

        prss and prss.report(1.0, "Correction completed")

        # 80–90%: Optional 2nd level generation
        second_topics = None
        if self.do_second_level:
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.8, 0.9) if prs else None
            prss and prss.report(0.0, "2nd-level topic generation")
            tg.generate_topic_lvl2(
                api=self.llm_provider,
                model=self.llm_model_type,
                seed_file=outputs['generation_topic'].as_posix(),
                data=path_sample.as_posix(),
                prompt_file=self._generation_2_prompt.as_posix(),
                out_file=outputs['generation_2_out'].as_posix(),
                topic_file=outputs['generation_2_topic'].as_posix(),
                verbose=self.verbose_scripts
            )
            second_topics = self._read_topics(outputs["generation_2_topic"])

        # 90–100%: Finalize topics and synthesize distributions
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.9, 1.0) if prs else None
        prss and prss.report(
            0.9, "Finalizing topics & synthetic distributions")

        topics = self._read_topics(outputs["generation_topic"])
        self.topics = topics
        print(f"Extracted {len(topics)} topics.")
        self.second_topics = second_topics

        # Save the raw topic strings
        with self.model_path.joinpath('orig_tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join([topics[k] for k in sorted(topics.keys())]))

        thetas, betas, vocab = self._approximate_distributions(assign_path=outputs['correction_out'], df=self.df)
        
        labels, summaries, add_info = self._format_results_by_topic()
        
        prss and prss.report(1.0, "Topics & synthetic distributions ready")

        return time.time() - t_start, thetas, betas, vocab, labels, summaries, add_info

    def infer_core(self, df_infer):
        """
        TopicGPT inference consists of re-using the assignment step.
        """

        # save infer data to temp file
        infer_path = pathlib.Path("topicgpt_infer_data.jsonl")
        df_infer_renamed = df_infer.rename(columns={"raw_text": "text"})
        df_infer_renamed.to_json(infer_path, lines=True, orient="records")

        # new topics should be saved in a different file
        assign_out_path = pathlib.Path("infer_assignment.jsonl")

        tg.assign_topics(
            api=self.llm_provider,
            model=self.llm_model_type,
            data=infer_path.as_posix(),
            prompt_file=self._assignment_prompt.as_posix(),
            topic_file=self.model_path.joinpath("modelFiles").joinpath(
                "generation_1.md").as_posix(),
            out_file=assign_out_path.as_posix(),
            verbose=self.verbose_scripts
        )

        # parse assignments to build thetas
        thetas = self._approximate_thetas(assign_out_path, df_infer)
        
        # remove temp files
        try:
            os.remove(infer_path)
            os.remove(assign_out_path)
        except Exception:
            pass
        
        return thetas, time.time() - 0

    def save_model(self):
        """
        Save TopicGPT model topics, metadata and the object itself.
        """
        model_p = self.model_path
        model_p.mkdir(parents=True, exist_ok=True)

        topics_txt = model_p.joinpath('topics.txt')
        topics2_txt = model_p.joinpath('topics_level2.txt')

        self._logger.info(f"Saving 1st-level topics to {topics_txt.as_posix()} and second-level topics to {topics2_txt.as_posix()}")

        if self.topics:
            with topics_txt.open('w', encoding='utf8') as f:
                for k in sorted(self.topics.keys()):
                    f.write(self.topics[k].rstrip('\n') + '\n')

        if self.second_topics:
            with topics2_txt.open('w', encoding='utf8') as f:
                for k in sorted(self.second_topics.keys()):
                    f.write(self.second_topics[k].rstrip('\n') + '\n')

        self._logger.info("Model saved successfully!")


    @classmethod
    def from_saved_model(cls, model_path: str):
        """
        Loads a previously saved TopicGPTTMmodel model from disk.

        Parameters
        ----------
        model_path : str
            Path to the saved model directory.
        Returns
        -------
        cls
            An instance of TopicGPTTMmodel with loaded topics and metadata.
        """
        
        obj = super().from_saved_model(model_path)
        model_p = pathlib.Path(model_path)

        topics_txt = model_p.joinpath('topics.txt')
        topics2_txt = model_p.joinpath('topics_level2.txt')

        # Load topics
        if topics_txt.exists():
            with topics_txt.open('r', encoding='utf8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            obj.topics = {i: ln for i, ln in enumerate(lines)}
            obj._logger.info(f"Loaded {len(lines)} topics from topics.txt")
        else:
            obj.topics = None
            obj._logger.warning("topics.txt not found; topics unavailable.")

        if topics2_txt.exists():
            with topics2_txt.open('r', encoding='utf8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            obj.second_topics = {i: ln for i, ln in enumerate(lines)}
            obj._logger.info(
                f"Loaded {len(lines)} 2nd-level topics from topics_level2.txt")
        else:
            obj.second_topics = None

        return obj

    def print_topics(self, verbose: bool = False, get_second_level: bool = False) -> dict:
        if get_second_level:
            if self.second_topics is None:
                self._logger.warning("Second-level topics not available.")
                return {}
            if verbose:
                for k, v in self.second_topics.items():
                    print(f"2nd-level Topic {k}: {v}")
            return self.second_topics or {}

        if self.topics is None:
            self._logger.warning("Topics not loaded yet.")
            return {}

        if verbose:
            for k, v in self.topics.items():
                print(f"Topic {k}: {v}")
        return self.topics or {}
