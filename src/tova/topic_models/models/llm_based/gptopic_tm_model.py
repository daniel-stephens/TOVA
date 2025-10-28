
import json
import logging
import os
import pathlib
import time
from subprocess import CalledProcessError, check_output
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize  # type: ignore
from tqdm import tqdm

from tova.topic_models.models.llm_based.base import LLMTModel
from tova.utils.cancel import CancellationToken, check_cancel
from tova.utils.progress import ProgressCallback  # type: ignore


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

        # Load TopicGPT parameters from config
        tg_cfg = self.config.get("topicgpt", {})
        # sampling
        self.sample = tg_cfg.get("sample", 0.001)
        # model names + decoding params
        self.deployment_name1 = tg_cfg.get("deployment_name1", "gpt-4")
        self.deployment_name2 = tg_cfg.get("deployment_name2", "gpt-3.5-turbo")
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

        # Paths to scripts/prompts
        cwd = pathlib.Path(os.getcwd())
        self._p_scripts = cwd / "src/topic_modeling/topicGPT/script"
        self._p_prompts = cwd / "src/topic_modeling/topicGPT/prompt"

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
            f"{self.__class__.__name__} initialized with num_topics={self.num_topics}, "
            f"sample={self.sample}, dep1='{self.deployment_name1}', dep2='{self.deployment_name2}', "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"do_second_level={self.do_second_level}."
        )

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

        path_sample = model_files / \
            f"sample_{str(self.sample)}.jsonl" if self.sample else model_files / \
            "sample.jsonl"
        df_sample.to_json(path_sample, lines=True, orient="records")
        self._logger.info(
            f"Sampling done. Using {len(df_sample)} docs. Saved to {path_sample.as_posix()}")

        prss and prss.report(1.0, "Workspace ready")

        # 5–25%: TOPIC GENERATION
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.05, 0.25) if prs else None
        prss and prss.report(0.0, "Topic generation")
        cmd = (
            f"python3 {self._p_scripts.joinpath('generation_1.py').as_posix()} "
            f"--deployment_name {self.deployment_name1} --max_tokens {self.max_tokens_gen1} "
            f"--temperature {self.temperature} --top_p {self.top_p} --data {path_sample.as_posix()} "
            f"--prompt_file {self._generation_prompt.as_posix()} --seed_file {self._seed_1.as_posix()} "
            f"--out_file {outputs['generation_out'].as_posix()} --topic_file {outputs['generation_topic'].as_posix()} "
            f"--verbose {self.verbose_scripts}"
        )
        self._run_cmd(cmd, "-- TOPIC GENERATION --")

        prss and prss.report(1.0, "Generation completed")

        # 25–45%: TOPIC REFINEMENT
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.25, 0.45) if prs else None
        prss and prss.report(0.0, "Topic refinement")
        cmd = (
            f"python3 {self._p_scripts.joinpath('refinement.py').as_posix()} "
            f"--deployment_name {self.deployment_name1} --max_tokens {self.max_tokens_assign} "
            f"--temperature {self.temperature} --top_p {self.top_p} "
            f"--prompt_file {self._refinement_prompt.as_posix()} "
            f"--generation_file {outputs['generation_out'].as_posix()} "
            f"--topic_file {outputs['generation_topic'].as_posix()} "
            f"--out_file {outputs['refinement_topic'].as_posix()} "
            f"--verbose {self.verbose_scripts} "
            f"--updated_file {outputs['refinement_out'].as_posix()} "
            f"--mapping_file {outputs['refinement_mapping'].as_posix()} "
            f"--refined_again {self.refined_again} --remove {self.remove}"
        )
        self._run_cmd(cmd, "-- TOPIC REFINEMENT --")
        prss and prss.report(1.0, "Refinement completed")

        # 45–65%: TOPIC ASSIGNMENT
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.45, 0.65) if prs else None
        prss and prss.report(0.0, "Topic assignment")
        cmd = (
            f"python3 {self._p_scripts.joinpath('assignment.py').as_posix()} "
            f"--deployment_name {self.deployment_name2} --max_tokens {self.max_tokens_assign} "
            f"--temperature {self.temperature} --top_p {self.top_p} "
            f"--data {path_sample.as_posix()} "
            f"--prompt_file {self._assignment_prompt.as_posix()} "
            f"--topic_file {outputs['generation_topic'].as_posix()} "
            f"--out_file {outputs['assignment_out'].as_posix()} "
            f"--verbose {self.verbose_scripts}"
        )
        self._run_cmd(cmd, "-- TOPIC ASSIGNMENT --")
        prss and prss.report(1.0, "Assignment completed")

        # 65–80%: TOPIC CORRECTION
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.65, 0.8) if prs else None
        prss and prss.report(0.0, "Topic correction")
        cmd = (
            f"python3 {self._p_scripts.joinpath('correction.py').as_posix()} "
            f"--deployment_name {self.deployment_name2} --max_tokens {self.max_tokens_assign} "
            f"--temperature {self.temperature} --top_p {self.top_p} "
            f"--data {outputs['assignment_out'].as_posix()} "
            f"--prompt_file {self._correction_prompt.as_posix()} "
            f"--topic_file {outputs['generation_topic'].as_posix()} "
            f"--out_file {outputs['correction_out'].as_posix()} "
            f"--verbose {self.verbose_scripts}"
        )
        self._run_cmd(cmd, "-- TOPIC CORRECTION --")
        prss and prss.report(1.0, "Correction completed")

        # 80–90%: Optional 2nd level generation
        second_topics = None
        if self.do_second_level:
            check_cancel(cancel, self._logger)
            prss = prs.report_subrange(0.8, 0.9) if prs else None
            prss and prss.report(0.0, "2nd-level topic generation")
            cmd = (
                f"python3 {self._p_scripts.joinpath('generation_2.py').as_posix()} "
                f"--deployment_name {self.deployment_name1} --max_tokens {self.max_tokens_gen2} "
                f"--temperature {self.temperature} --top_p {self.top_p} "
                f"--data {outputs['generation_out'].as_posix()} "
                f"--seed_file {outputs['generation_topic'].as_posix()} "
                f"--prompt_file {self._generation_2_prompt.as_posix()} "
                f"--out_file {outputs['generation_2_out'].as_posix()} "
                f"--topic_file {outputs['generation_2_topic'].as_posix()} "
                f"--verbose {self.verbose_scripts}"
            )
            self._run_cmd(cmd, "-- 2nd LEVEL TOPIC GENERATION --")
            second_topics = self._read_topics(outputs["generation_2_topic"])

        # 90–100%: Finalize topics and synthesize distributions
        check_cancel(cancel, self._logger)
        prss = prs.report_subrange(0.9, 1.0) if prs else None
        prss and prss.report(
            0.9, "Finalizing topics & synthetic distributions")

        topics = self._read_topics(outputs["generation_topic"])
        self.topics = topics
        self.second_topics = second_topics

        # ---- Synthesize thetas/betas/vocab so TradTMmodel can proceed ----
        # Vocab: from lemmas in training set
        vocab = sorted({w for doc in self.train_data for w in doc})
        V = max(len(vocab), 1)
        T = max(len(topics), 1)
        D = max(len(self.train_data), 1)

        # Uniform betas (T x V) and thetas (D x T); then normalized
        betas = np.full((T, V), 1.0 / V, dtype=float)
        thetas = np.full((D, T), 1.0 / T, dtype=float)
        thetas = normalize(thetas, axis=1, norm='l1')

        # Save the raw topic strings
        with self.model_path.joinpath('orig_tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join([topics[k] for k in sorted(topics.keys())]))

        prss and prss.report(1.0, "Topics & synthetic distributions ready")

        return time.time() - t_start, thetas, betas, vocab


    def infer_core(self, infer_data, df_infer, embeddings_infer):
        """
        TopicGPT doesn't support inference.
        """
        raise RuntimeError(
            "TopicGPT does not support inference in this pipeline.")

    def save_model(self):
        """
        Save TopicGPT artifacts: topics, second-level topics (if any), and config.
        """
        model_p = self.model_path
        topics_txt = model_p.joinpath('topics.txt')
        topics2_txt = model_p.joinpath('topics_level2.txt')
        meta_json = model_p.joinpath('topicgpt_meta.json')

        self._logger.info(f"Saving TopicGPT topics to {topics_txt.as_posix()}")
        if self.topics:
            with topics_txt.open('w', encoding='utf8') as f:
                for k in sorted(self.topics.keys()):
                    f.write(self.topics[k].rstrip('\n') + '\n')

        if self.second_topics:
            self._logger.info(
                f"Saving 2nd-level topics to {topics2_txt.as_posix()}")
            with topics2_txt.open('w', encoding='utf8') as f:
                for k in sorted(self.second_topics.keys()):
                    f.write(self.second_topics[k].rstrip('\n') + '\n')

        meta = dict(
            num_topics=self.num_topics,
            sample=self.sample,
            deployment_name1=self.deployment_name1,
            deployment_name2=self.deployment_name2,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens_gen1=self.max_tokens_gen1,
            max_tokens_gen2=self.max_tokens_gen2,
            max_tokens_assign=self.max_tokens_assign,
            refined_again=self.refined_again,
            remove=self.remove,
            do_second_level=self.do_second_level,
            verbose=self.verbose_scripts,
        )
        self._logger.info(
            f"Saving TopicGPT metadata to {meta_json.as_posix()}")
        with meta_json.open('w', encoding='utf8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self._logger.info("TopicGPT artifacts saved successfully!")

    @classmethod
    def from_saved_model(cls, model_path: str):
        """
        Restore TopicGPTTMmodel from saved topics & metadata.
        """
        obj = cls(model_path=model_path, load_model=True)
        model_p = pathlib.Path(model_path)

        topics_txt = model_p.joinpath('topics.txt')
        topics2_txt = model_p.joinpath('topics_level2.txt')
        meta_json = model_p.joinpath('topicgpt_meta.json')

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

        # Load metadata (optional)
        if meta_json.exists():
            with meta_json.open('r', encoding='utf8') as f:
                meta = json.load(f)
            for k, v in meta.items():
                setattr(obj, k, v)
            obj._logger.info("Loaded TopicGPT metadata.")

        return obj

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
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
