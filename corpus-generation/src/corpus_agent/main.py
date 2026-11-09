from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

from corpus_state.db import create_database, list_instances

from corpus_agent.config import load_config
from corpus_agent.id_allocator import InstanceIdAllocator
from corpus_agent.llm import LLMClient
from corpus_agent.phases.character_mapping import run_character_mapping
from corpus_agent.phases.email_generator import generate_email
from corpus_agent.phases.export import export_corpus
from corpus_agent.phases.instance_generation import run_instance
from corpus_agent.phases.question_extraction import run_question_extraction
from corpus_agent.phases.question_planning import (
    format_plan_for_generation,
    load_question_plan,
    plan_study_questions,
    save_question_plan,
)
from corpus_agent.phases.story_resolution import resolve_story_mapping
from corpus_agent.phases.wave_planner import plan_wave
from corpus_agent.sizing import resolve_corpus_size
from corpus_agent.skills import load_all_skills

logger = logging.getLogger(__name__)


async def run_pipeline(
    config_path: str,
    resume: bool = False,
    skills_root: str | None = None,
    extract_questions: bool = False,
    questions_only: bool = False,
) -> None:
    cfg = load_config(config_path)
    conn = create_database(cfg.output.db_path, check_same_thread=False)

    load_kwargs = {}
    if skills_root:
        load_kwargs["skills_root"] = skills_root
    skills = load_all_skills(**load_kwargs)

    api_key = os.getenv(cfg.llm.api_key_env_var)
    if not api_key:
        raise ValueError(
            f"Environment variable {cfg.llm.api_key_env_var} is not set"
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=cfg.llm.base_url,
    )
    llm = LLMClient(
        async_client=client,
        rate_limits=cfg.rate_limits,
        max_tokens=cfg.llm.max_tokens,
    )

    logger.info("Phase -1: Story resolution for corpus %s", cfg.corpus_id)
    plot_outline = await resolve_story_mapping(
        conn=conn,
        corpus_id=cfg.corpus_id,
        story_mapping=cfg.story_mapping,
        source_story_path=cfg.source_story.path,
        blur_level=cfg.blur_level,
        llm=llm,
        model=cfg.models.outline_generator,
        skill_content=skills["outline-generation"],
    )
    logger.info("Phase -1 complete: %d plot points resolved", len(plot_outline))

    resolved_size = resolve_corpus_size(
        corpus_size=cfg.corpus_size,
        num_plot_points=len(plot_outline),
        blur_level=cfg.blur_level,
        mode=cfg.story_mapping.plot_outline.mode,
    )
    total_target = cfg.corpus_size.total_emails
    if total_target is not None:
        logger.info(
            "Corpus sizing resolved: %d total_emails -> %d instances, %d waves, %d epw, %d cap",
            total_target,
            resolved_size.num_instances,
            resolved_size.waves_per_instance,
            resolved_size.emails_per_wave,
            resolved_size.emails_per_instance,
        )
    else:
        logger.info(
            "Corpus sizing resolved: %d instances, %d waves, %d epw, %d cap",
            resolved_size.num_instances,
            resolved_size.waves_per_instance,
            resolved_size.emails_per_wave,
            resolved_size.emails_per_instance,
        )

    logger.info("Phase 0: Character mapping for corpus %s", cfg.corpus_id)
    characters = await run_character_mapping(
        conn=conn,
        llm=llm,
        model=cfg.models.character_mapping,
        corpus_id=cfg.corpus_id,
        source_story_path=cfg.source_story.path,
        setting=cfg.story_mapping.synthetic_setting,
        blur_level=cfg.blur_level,
        plot_outline=plot_outline,
        skill_content=skills["character-mapping"],
    )
    logger.info("Phase 0 complete: %d characters mapped", len(characters))

    question_plan_path = Path(cfg.output.db_path).parent / "question_plans" / f"{cfg.corpus_id}.json"
    question_plan = load_question_plan(question_plan_path)
    if question_plan is None:
        logger.info("Phase 0.5: Planning study questions before any documents")
        question_plan = await plan_study_questions(
            llm=llm,
            model=cfg.models.question_extractor,
            corpus_id=cfg.corpus_id,
            characters=characters,
            plot_outline=plot_outline,
            content_style=cfg.content_style,
        )
        save_question_plan(question_plan, question_plan_path)
        logger.info(
            "Phase 0.5 complete: %d questions written to %s",
            len(question_plan.questions),
            question_plan_path,
        )
    else:
        logger.info(
            "Phase 0.5: Reusing %d planned questions from %s",
            len(question_plan.questions),
            question_plan_path,
        )
    question_brief = format_plan_for_generation(question_plan)

    if not questions_only:
        if resume:
            in_progress = list_instances(
                conn, status="in_progress", corpus_id=cfg.corpus_id
            )
            instance_ids = [inst.instance_id for inst in in_progress]
            remaining_new = (
                resolved_size.num_instances
                - len(instance_ids)
                - len(list_instances(conn, status="completed", corpus_id=cfg.corpus_id))
            )
            if remaining_new > 0:
                alloc = InstanceIdAllocator(conn, cfg.corpus_id, remaining_new)
                instance_ids.extend(alloc.next_id() for _ in range(remaining_new))
        else:
            alloc = InstanceIdAllocator(conn, cfg.corpus_id, resolved_size.num_instances)
            instance_ids = [alloc.next_id() for _ in range(resolved_size.num_instances)]

        logger.info("Phase 1: Generating %d instances", len(instance_ids))
        sem = asyncio.Semaphore(cfg.rate_limits.max_concurrent_instances)

        async def _run_one(iid: int) -> None:
            async with sem:

                async def _plan_wave(**kwargs):
                    return await plan_wave(
                        llm=llm,
                        model=cfg.models.planner,
                        skill_content=skills["orchestration"],
                        content_style=cfg.content_style,
                        question_brief=question_brief,
                        **kwargs,
                    )

                async def _generate_email(**kwargs):
                    return await generate_email(
                        llm=llm,
                        model=cfg.models.content_generator,
                        skill_content=skills["email-generation"],
                        content_style=cfg.content_style,
                        question_brief=question_brief,
                        **kwargs,
                    )

                await run_instance(
                    conn=conn,
                    instance_id=iid,
                    corpus_id=cfg.corpus_id,
                    characters=characters,
                    plot_outline=plot_outline,
                    waves_per_instance=resolved_size.waves_per_instance,
                    emails_per_wave=resolved_size.emails_per_wave,
                    emails_per_instance=resolved_size.emails_per_instance,
                    blur_level=cfg.blur_level,
                    plan_wave_fn=_plan_wave,
                    generate_email_fn=_generate_email,
                    wave_pause_seconds=cfg.rate_limits.wave_pause_seconds,
                    question_plan=question_plan,
                    export_path=cfg.output.export_path,
                )

        results = await asyncio.gather(
            *[_run_one(iid) for iid in instance_ids], return_exceptions=True
        )
        for iid, result in zip(instance_ids, results):
            if isinstance(result, Exception):
                logger.error("Instance %d failed with unhandled exception: %s", iid, result)
        logger.info("Phase 1 complete")

        logger.info("Phase 2: Exporting corpus")
        export_corpus(conn, cfg.output.export_path, cfg.corpus_id)
        logger.info("Phase 2 complete: exported to %s", cfg.output.export_path)

    # Phase 3: Question extraction (--extract-questions or --questions-only)
    if extract_questions or questions_only:
        logger.info("Phase 3: Extracting sensemaking questions")
        question_sets = await run_question_extraction(
            conn=conn,
            llm=llm,
            model=cfg.models.question_extractor,
            corpus_id=cfg.corpus_id,
            characters=characters,
            plot_outline=plot_outline,
            skill_content=skills["question-extraction"],
            export_path=cfg.output.export_path,
            content_style=cfg.content_style,
        )
        logger.info(
            "Phase 3 complete: %d instances processed, questions exported to %s",
            len(question_sets),
            cfg.output.export_path,
        )

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic email corpus")
    parser.add_argument("--config", required=True, help="Path to corpus_config.yaml")
    parser.add_argument(
        "--resume", action="store_true", help="Resume incomplete instances"
    )
    parser.add_argument(
        "--extract-questions",
        action="store_true",
        help="Extract sensemaking questions after corpus generation",
    )
    parser.add_argument(
        "--questions-only",
        action="store_true",
        help="Skip generation and export; extract questions from an already-complete corpus",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(
        run_pipeline(
            args.config,
            resume=args.resume,
            extract_questions=args.extract_questions,
            questions_only=args.questions_only,
        )
    )


if __name__ == "__main__":
    main()
