"""Study group feature flags and preset model access for user studies."""

from __future__ import annotations

from django.contrib.auth.models import AnonymousUser

from web.models import User

FEATURE_ACTIVE_LEARNING = "active_learning"

FEATURE_FIELDS = {
    FEATURE_ACTIVE_LEARNING: "enable_active_learning",
}


def _resolve_user(user) -> User | None:
    if not user or isinstance(user, AnonymousUser) or not user.is_authenticated:
        return None
    if not isinstance(user, User):
        return None
    if getattr(user, "study_group_id", None) and getattr(user, "study_group", None) is None:
        user = User.objects.select_related("study_group").filter(pk=user.pk).first()
    return user if isinstance(user, User) else None


def is_study_participant(user) -> bool:
    """A user is a participant once they have a dashboard assigned.

    Membership of a StudyGroup also counts, so cohorts created before per-user
    assignment existed keep working.
    """
    user = _resolve_user(user)
    if not user:
        return False
    return bool((user.study_model_id or "").strip() or user.study_group_id)


def get_study_group(user):
    user = _resolve_user(user)
    if not user:
        return None
    return user.study_group


def get_study_model_id(user) -> str | None:
    """The dashboard assigned to *user*: their own, else their group's."""
    user = _resolve_user(user)
    if not user:
        return None
    own = (user.study_model_id or "").strip()
    if own:
        return own
    group = user.study_group
    if group:
        inherited = (group.default_model_id or "").strip()
        if inherited:
            return inherited
    return None


def user_has_feature(user, feature: str) -> bool:
    """Return True if *user* may use *feature* (non-study users get full access)."""
    user = _resolve_user(user)
    if not user:
        return False
    if not is_study_participant(user):
        return True

    field = FEATURE_FIELDS.get(feature)
    if not field:
        return True

    if feature == FEATURE_ACTIVE_LEARNING:
        # A run already underway keeps the condition it started in, so an admin
        # flipping the switch mid-session cannot split one participant's data
        # across two conditions. This has to be the single source of truth --
        # template gating, view decorators and the flow middleware all read it,
        # and they must agree or the UI offers a tab the route then refuses.
        snapshot = _run_snapshot(user)
        if snapshot is not None:
            return bool(snapshot)

        # Per-user setting wins; None means "inherit".
        own = user.enable_active_learning
        if own is not None:
            return bool(own)

    group = user.study_group
    if group is None:
        # Assigned a dashboard but no group: active learning is on by default.
        return True
    return bool(getattr(group, field, True))



def _run_snapshot(user) -> bool | None:
    """Active-learning state frozen when this user's run began, if it has.

    Returns None when there is no started run, so callers fall back to the live
    setting. Cached on the user instance: this is called once per request per
    template render.
    """
    cached = getattr(user, "_al_snapshot_cache", "unset")
    if cached != "unset":
        return cached

    value = None
    try:
        from web.models import ParticipantSession

        row = (
            ParticipantSession.objects.filter(pk=user.pk)
            .exclude(task_started_at=None)
            .values_list("active_learning", flat=True)
            .first()
        )
        if row is not None:
            value = bool(row)
    except Exception:
        value = None

    try:
        user._al_snapshot_cache = value
    except Exception:
        pass
    return value

def get_user_features(user) -> dict[str, bool]:
    """Feature flags for templates and JSON responses."""
    return {
        FEATURE_ACTIVE_LEARNING: user_has_feature(user, FEATURE_ACTIVE_LEARNING),
    }


def user_can_access_model(user, model_id: str) -> bool:
    """True if *model_id* is the preset model for the user's study group."""
    if not model_id:
        return False
    preset = get_study_model_id(user)
    return bool(preset and str(model_id) == str(preset))


def study_participant_may_access_model(user, model_id: str) -> bool:
    """Study participants may only use their group's preset model."""
    if not is_study_participant(user):
        return True
    if not model_id:
        return False
    return user_can_access_model(user, model_id)


def post_login_redirect_url(user) -> str:
    """Return the URL path participants should land on after login.

    Delegates to the study flow so login and mid-flow redirects can never
    disagree about where a participant belongs.
    """
    from urllib.parse import urlencode

    from django.urls import reverse

    if not is_study_participant(user):
        return reverse("web:home")

    # Imported lazily to avoid a circular import (study_flow imports this module).
    from web.study_flow import get_or_create_session, stage_redirect_url

    try:
        return stage_redirect_url(get_or_create_session(user))
    except Exception:  # pragma: no cover - fall back to the dashboard
        model_id = get_study_model_id(user)
        if model_id:
            return reverse("web:dashboard") + "?" + urlencode({"model_id": model_id})
        return reverse("web:home")
