"""Study flow state machine.

Participants move monotonically through:

    consent -> instructions -> tutorial -> task -> post_study -> done

The tables below declare which named routes are reachable at each stage;
``StudyFlowMiddleware`` enforces them. Keeping the tables here (rather than
inline in the middleware) means adding a route to the task stage is a one-line
data change.

Staff, superusers and TOVA app admins are never gated -- that check lives in the
middleware, not here.
"""

from __future__ import annotations

import datetime as _dt

from django.conf import settings
from django.urls import reverse
from django.utils import timezone

from web.models import ParticipantSession, StudyResponse
from web.study_features import (
    FEATURE_ACTIVE_LEARNING,
    get_study_model_id,
    user_has_feature,
)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

#: Minutes a participant gets to explore before the task auto-closes.
DEFAULT_TASK_MINUTES = 30


def task_minutes(subject=None) -> int:
    """Exploration minutes for *subject*, falling back to the global default.

    Resolution order mirrors the tri-state used for active learning: the
    participant's own override, then their group's, then ``STUDY_TASK_MINUTES``.

    The approved protocol specifies a fixed 30-minute period, so the default is
    what a real run uses -- the overrides exist for piloting, and to let a later
    study vary exposure per condition without another schema change.
    """
    default = int(getattr(settings, "STUDY_TASK_MINUTES", DEFAULT_TASK_MINUTES))
    if subject is None:
        return default

    own = getattr(subject, "task_minutes", None)
    if own:
        return int(own)

    group = getattr(subject, "study_group", None)
    inherited = getattr(group, "task_minutes", None) if group else None
    if inherited:
        return int(inherited)
    return default


# ---------------------------------------------------------------------------
# Informed consent
# ---------------------------------------------------------------------------

#: Identifies the consent text a participant was shown, stamped onto their
#: session when they consent.
#:
#: The study runs under UMD IRB package 1936243-1 on *implied* consent: the
#: participant consents by clicking through, and there is no signed form. That
#: makes the rendered text the consent instrument itself, so it is reproduced
#: verbatim in ``templates/study/consent.html`` -- never paraphrased, summarised
#: or reordered. Bump this constant in the same commit as any change to that
#: wording, so the export always says which version each participant agreed to.
#:
#: Source: "TOVA CorpusBio - consent document revised.docx", with the
#: investigator's telephone number removed at the research team's request. That is a
#: departure from the approved document, which lists name, address, email and
#: phone -- hence the version bump, so anyone who consented to the earlier text
#: is still identifiable in the export.
CONSENT_VERSION = "corpusbio-2026-08-31-no-phone"

#: IRB identifiers shown to participants and carried into the export.
IRB_PACKAGE_NUMBER = "1936243-1"
IRB_CONTACT_EMAIL = "irb@umd.edu"


# ---------------------------------------------------------------------------
# Post-study instrument
# ---------------------------------------------------------------------------

# The instrument itself lives in web.instruments so that revising the questions
# is a versioned change rather than an invisible one. Re-exported here because
# the views, the admin export and the tests have always reached for
# study_flow.TF_KEY and there is no value in churning those call sites.
from web.instruments import (  # noqa: F401  (re-export)
    INSTRUMENT_VERSION,
    OPEN_QUESTIONS,
    TF_DECLINED,
    TF_KEY,
    TF_VALID_VALUES,
    count_tf_answered,
    score_tf,
)

#: Label for documents whose theme could not be resolved. Matches the
#: ``'Other'`` fallback already present in the picker's ``updateTrigger``.
UNCLASSIFIED_THEME = "Unclassified"


# ---------------------------------------------------------------------------
# Route tables
# ---------------------------------------------------------------------------

#: Reachable at any stage.
ALWAYS_ALLOWED = {
    "logout",
    "login",
    "study_login",
    "login_okta",
    "auth_okta_callback",
    "terms",
    "privacy",
    "django_health",
    "check_backend",
}

#: Everything the dashboard needs in order to function during the task.
_TASK_URL_NAMES = {
    "study_task",
    # Re-readable while exploring. Time spent here is charged like any other
    # request: it is part of the 30 minutes, not a way to pause them.
    "study_tutorial",
    "study_timer",
    "study_finish",
    "study_events",
    "dashboard",
    "get_dashboard_data",
    "text_info",
    "infer_text",
    "get_corpus",
    "getAllCorpora",
    "get_unique_corpus_names",
    "get_status_jobs",
    "get_status_root",
    "save_settings",
    "api_user_config",
    "api_get_user_config_overrides",
    "rename_topic",
    "get_topic_renames",
    # Dashboard theme tools. The middleware is fail-closed, so a view added
    # without being listed here is refused for participants and the control
    # silently does nothing -- which is what happened to these three.
    "suggest_theme_label",
    "set_document_theme",
    "get_document_themes",
    "chat",
    "chat_stream",
    "get_chat_messages",
    "chat_llm_options",
    "chat_openai_key_status",
    "suggest_doc_topics",
}

#: Unioned into the task stage only when the group has active learning enabled.
_AL_URL_NAMES = {
    "active_learning_two",
    "active_learning_three",
    "active_learning_two_recommend",
    "active_learning_two_status",
    "active_learning_two_label",
    "active_learning_two_unlabel",
    "active_learning_two_labels",
    "active_learning_two_bootstrap",
}

STAGE_URL_NAMES = {
    # Nothing but the consent page itself is reachable before consenting -- not
    # the instructions, and not the tool.
    ParticipantSession.STAGE_CONSENT: {
        "study_consent",
        "study_consent_accept",
        "study_consent_declined",
        "study_events",
    },
    ParticipantSession.STAGE_INSTRUCTIONS: {"study_instructions", "study_tutorial", "study_events"},
    # The walkthrough is where the task is actually started, so study_begin
    # lives here rather than on the instructions page.
    ParticipantSession.STAGE_TUTORIAL: {"study_tutorial", "study_begin", "study_events"},
    ParticipantSession.STAGE_TASK: _TASK_URL_NAMES,
    # study_timer/study_finish stay reachable after the task so the countdown
    # widget gets a clean JSON hand-off when the deadline passes, rather than
    # discovering the stage change via a failed parse. Both are no-ops here.
    ParticipantSession.STAGE_POST: {"study_post", "study_docs", "study_timer", "study_finish", "study_events"},
    ParticipantSession.STAGE_DONE: {"study_done", "study_timer", "study_events"},
}


def allowed_url_names(session: ParticipantSession, user=None) -> set:
    """Route names reachable at *session*'s current stage.

    Active learning is read from the session snapshot rather than the live user
    flag: once a run has started, flipping the switch must not change the
    condition the participant is already in.
    """
    names = set(STAGE_URL_NAMES.get(session.stage, set()))
    if session.stage == ParticipantSession.STAGE_TASK and session.active_learning:
        names |= _AL_URL_NAMES
    return names


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def get_or_create_session(user) -> ParticipantSession:
    """Return the participant's session, creating it at the consent stage."""
    session = ParticipantSession.objects.filter(pk=user.pk).select_related("study_group").first()
    if session is not None:
        return session
    session = ParticipantSession.objects.create(
        user=user,
        study_group=user.study_group,
        model_id=get_study_model_id(user) or "",
        active_learning=user_has_feature(user, FEATURE_ACTIVE_LEARNING),
    )
    return session


def begin_task(session: ParticipantSession, user=None) -> ParticipantSession:
    """Start the exploration timer. Idempotent -- re-entry never extends it."""
    if session.task_started_at is not None:
        return session

    now = timezone.now()
    subject = user if user is not None else session.user
    session.study_group = subject.study_group
    session.model_id = get_study_model_id(subject) or session.model_id
    session.active_learning = user_has_feature(subject, FEATURE_ACTIVE_LEARNING)
    session.task_minutes = task_minutes(subject)
    session.task_started_at = now
    session.task_elapsed_s = 0
    session.last_seen_at = now
    session.task_ends_at = now + _dt.timedelta(minutes=session.task_minutes)
    session.advance_to(ParticipantSession.STAGE_TASK)
    session.save(
        update_fields=[
            "study_group",
            "model_id",
            "active_learning",
            "task_minutes",
            "task_elapsed_s",
            "last_seen_at",
            "task_started_at",
            "task_ends_at",
            "stage",
        ]
    )
    return session


def touch_task(session: ParticipantSession) -> ParticipantSession:
    """Charge the time since this participant's last request, and mark them seen.

    This is what makes the clock pause. Presence is inferred from requests: a
    gap within ``PRESENCE_MAX_GAP_S`` is time they were here and is charged; a
    longer gap means they had signed out, closed the tab or lost connection, and
    costs them nothing.

    Called from the flow middleware, so every request a participant makes during
    the task counts -- not only the timer widget's poll. A participant with
    JavaScript disabled still accrues time by using the tool.
    """
    if session.stage != ParticipantSession.STAGE_TASK or not session.task_started_at:
        return session

    now = timezone.now()
    charged = 0
    gap = None
    if session.last_seen_at:
        gap = (now - session.last_seen_at).total_seconds()
        if 0 < gap <= ParticipantSession.PRESENCE_MAX_GAP_S:
            charged = int(gap)

    # Skip the write for rapid successive requests: a page load can fire several
    # at once and there is no point in an UPDATE per asset.
    if gap is not None and 0 <= gap < 1:
        return session

    session.task_elapsed_s = int(session.task_elapsed_s or 0) + charged
    session.last_seen_at = now
    session.task_ends_at = now + _dt.timedelta(seconds=session.seconds_remaining)
    session.save(update_fields=["task_elapsed_s", "last_seen_at", "task_ends_at"])
    return session


def start_charging_now(session: ParticipantSession) -> ParticipantSession:
    """Begin charging presence from this moment, discarding the wait before it.

    The clock starts when the participant presses "Start exploring", but the
    tool is not usable until the dashboard has its data, several seconds later.
    Charging that wait would be unfair, and worse, it would not be constant:
    different model backbones return data at different speeds, so the loss
    would line up exactly with the study's condition and nothing in the data
    would show it.

    Called once, when the dashboard's data first arrives. ``task_started_at``
    is left alone, so the record still says when they pressed the button.
    """
    if session.stage != ParticipantSession.STAGE_TASK:
        return session
    if not session.task_started_at or session.tool_ready_at:
        # Not running, or the tool has already been ready once. Guarded on its
        # own timestamp rather than on elapsed time: the flow middleware charges
        # every request, including the ones the page load itself makes, so by
        # the time this runs the wait has already been counted. Checking
        # "elapsed is still zero" would therefore never be true and this would
        # silently do nothing.
        return session

    now = timezone.now()
    session.tool_ready_at = now
    session.task_elapsed_s = 0          # refund what the load already cost
    session.last_seen_at = now
    session.task_ends_at = now + _dt.timedelta(seconds=session.task_total_seconds)
    session.save(update_fields=[
        "tool_ready_at", "task_elapsed_s", "last_seen_at", "task_ends_at",
    ])
    return session


def finish_task(session: ParticipantSession, *, early: bool) -> ParticipantSession:
    """Close the exploration task and move the participant to the questions."""
    if session.stage != ParticipantSession.STAGE_TASK:
        return session

    # Charge the stretch since their last request before closing, or the final
    # seconds of exploration would go unrecorded.
    touch_task(session)

    session.task_completed_at = timezone.now()
    session.finished_early = bool(early)
    session.advance_to(ParticipantSession.STAGE_POST)
    session.save(update_fields=["task_completed_at", "finished_early", "stage"])
    return session


def complete_study(session: ParticipantSession) -> ParticipantSession:
    session.completed_at = timezone.now()
    session.advance_to(ParticipantSession.STAGE_DONE)
    session.save(update_fields=["completed_at", "stage"])
    return session


def stage_redirect_url(session: ParticipantSession) -> str:
    """Where a participant at *session*'s stage belongs."""
    if session.stage == ParticipantSession.STAGE_TASK:
        return reverse("web:study_task")
    if session.stage == ParticipantSession.STAGE_POST:
        return reverse("web:study_post")
    if session.stage == ParticipantSession.STAGE_DONE:
        return reverse("web:study_done")
    if session.stage == ParticipantSession.STAGE_TUTORIAL:
        return reverse("web:study_tutorial")
    if session.stage == ParticipantSession.STAGE_INSTRUCTIONS:
        return reverse("web:study_instructions")
    return reverse("web:study_consent")


def record_consent(session: ParticipantSession) -> ParticipantSession:
    """Record implied consent and open the instructions. Idempotent.

    Re-POSTing never restamps the timestamp: the first click is the one that
    happened, and an accidental double-submit must not rewrite the record.
    """
    if session.consented_at is not None:
        return session

    session.consented_at = timezone.now()
    session.consent_version = CONSENT_VERSION
    session.advance_to(ParticipantSession.STAGE_INSTRUCTIONS)
    session.save(update_fields=["consented_at", "consent_version", "stage"])
    return session


def has_submitted(user) -> bool:
    return StudyResponse.objects.filter(pk=user.pk).exists()
