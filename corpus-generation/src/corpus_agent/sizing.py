from __future__ import annotations

import logging
import math

from pydantic import BaseModel

from corpus_agent.config import CorpusSize

logger = logging.getLogger(__name__)

_EMAILS_PER_WAVE_BY_BLUR = {"light": 3, "medium": 4, "heavy": 5}

_CAP_MULTIPLIER = 1.5


class ResolvedCorpusSize(BaseModel):
    num_instances: int
    waves_per_instance: int
    emails_per_wave: int
    emails_per_instance: int


def resolve_corpus_size(
    corpus_size: CorpusSize,
    num_plot_points: int,
    blur_level: str,
    mode: str,
) -> ResolvedCorpusSize:
    explicitly_set = corpus_size.model_fields_set

    if mode == "auto":
        waves = (
            corpus_size.waves_per_instance
            if "waves_per_instance" in explicitly_set
            else num_plot_points
        )
        epw = (
            corpus_size.emails_per_wave
            if "emails_per_wave" in explicitly_set
            else _EMAILS_PER_WAVE_BY_BLUR[blur_level]
        )
        cap = (
            corpus_size.emails_per_instance
            if "emails_per_instance" in explicitly_set
            else int(waves * epw * _CAP_MULTIPLIER)
        )
        if "num_instances" in explicitly_set:
            num_inst = corpus_size.num_instances
        elif corpus_size.total_emails is not None:
            num_inst = math.ceil(corpus_size.total_emails / cap)
        else:
            raise ValueError(
                "Auto mode must provide either 'total_emails' or explicit "
                "'num_instances' in corpus_size"
            )
    else:
        if corpus_size.num_instances is None:
            raise ValueError(
                "num_instances is required in manual mode"
            )
        if corpus_size.total_emails is not None:
            logger.warning(
                "total_emails=%d is ignored in manual mode; "
                "num_instances=%d will be used",
                corpus_size.total_emails,
                corpus_size.num_instances,
            )
        waves = corpus_size.waves_per_instance
        epw = corpus_size.emails_per_wave
        cap = corpus_size.emails_per_instance
        num_inst = corpus_size.num_instances

    return ResolvedCorpusSize(
        num_instances=num_inst,
        waves_per_instance=waves,
        emails_per_wave=epw,
        emails_per_instance=cap,
    )
