"""The post-study instrument: questions, answer key, and scoring.

Separated from the flow logic so the questions can be revised without touching
the state machine, and -- more importantly -- so a revision is *versioned*.

``tf_score`` is computed at submission time against whatever the key said then.
If the wording or the key ever changes mid-study, scores from before and after
are not comparable, and nothing in the stored row would say so. Stamping
:data:`INSTRUMENT_VERSION` onto each response makes that visible in the export
instead of silently corrupting the analysis.

Bump the version whenever the statements, the questions, or the key change.

The instrument deliberately lives in code rather than in the database. A
DB-backed editor is a much larger build (permissions, drafts, preview, migration
of in-flight responses) and the questions here are fixed by the approved
protocol, so the thing that actually matters is reproducible scoring, not
editability.
"""

from __future__ import annotations

#: Identifies the instrument a response was scored against.
INSTRUMENT_VERSION = "corpusbio-v1"

#: Correct answers for the true/false statements, keyed by field name.
#: Server-side deliberately -- this used to sit in the DOM as ``data-answer``
#: attributes, where a participant could read it from view-source.
TF_KEY = {
    "tf1": "true",
    "tf2": "false",
    "tf3": "true",
    "tf4": "false",
}

#: Explicitly declining to answer. The approved consent says "You may skip a
#: question", so this is a first-class response rather than a validation error.
#: Kept distinct from ``unsure``: "I would rather not say" and "I don't know"
#: are different findings, and collapsing them would lose that.
TF_DECLINED = "declined"

TF_VALID_VALUES = {"true", "false", "unsure", TF_DECLINED}

#: Free-text questions: field name -> the category label shown on the page.
OPEN_QUESTIONS = {
    "motive": "Motive",
    "connections": "Connections",
    "timeline": "Timeline",
    "threads": "Distinguishing threads",
}


def score_tf(answers: dict) -> int:
    """Number of true/false statements answered correctly.

    ``unsure`` is not correct, but it is stored verbatim so the distinction
    between "wrong" and "didn't know" survives into analysis.
    """
    return sum(1 for key, correct in TF_KEY.items() if answers.get(key) == correct)


def count_tf_answered(answers: dict) -> int:
    """How many statements got a substantive response.

    Read alongside :func:`score_tf`: a low score because someone skipped six
    questions is a different result from a low score because they were wrong,
    and without this the two are indistinguishable in the export. Skipped
    (absent) and ``declined`` both count as unanswered; ``unsure`` does not,
    because saying so is a real answer.
    """
    return sum(1 for key in TF_KEY if answers.get(key) not in (None, "", TF_DECLINED))
