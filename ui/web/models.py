"""TOVA UI database models (greenfield Django ORM)."""

from __future__ import annotations

import uuid

from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("email required")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_admin", True)
        return self.create_user(email, password, **extra_fields)


class StudyGroup(models.Model):
    """Study cohort with a preset model and per-group feature toggles."""

    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    default_model_id = models.CharField(
        max_length=255,
        blank=True,
        help_text="Trained model ID all users in this group land on after login.",
    )
    enable_active_learning = models.BooleanField(default=True)
    task_minutes = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text=(
            "Exploration minutes for this condition. Leave blank to use the "
            "site default (30, as the protocol specifies)."
        ),
    )

    class Meta:
        db_table = "web_study_groups"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class User(AbstractUser):
    """Email-based user (no username)."""

    username = None
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField("email address", unique=True)
    name = models.CharField(max_length=255, blank=True)
    auth_source = models.CharField(max_length=50, blank=True)
    is_admin = models.BooleanField(default=False)
    study_group = models.ForeignKey(
        StudyGroup,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="users",
    )
    # The dashboard (trained model) this participant works with. This is the
    # source of truth: the study "grouping" is simply everyone on the same
    # dashboard. StudyGroup remains an optional label for naming a condition.
    study_model_id = models.CharField(
        max_length=255,
        blank=True,
        verbose_name="assigned dashboard",
        help_text="Trained model backing this user's dashboard. Assigning one makes them a study participant.",
    )
    # The study-specific code this participant is known by in the analytic data.
    # The approved consent says "Data will be linked only to a study-specific
    # code", and that a recruitment platform's account information "will not be
    # copied into the analytic dataset" -- so exports key on this and never on
    # email. Generated accounts get it automatically (p001@study.local -> p001);
    # it stays blank for staff and for anyone invited by real email address.
    participant_code = models.CharField(
        max_length=64,
        blank=True,
        null=True,
        unique=True,
        help_text=(
            "Study code used in exported data. Set automatically for generated "
            "participant logins."
        ),
    )
    task_minutes = models.PositiveIntegerField(
        null=True,
        blank=True,
        verbose_name="exploration minutes",
        help_text="Overrides the group and the site default. Blank inherits.",
    )
    # Tri-state: None inherits from the study group (or defaults to on).
    enable_active_learning = models.BooleanField(
        null=True,
        blank=True,
        default=None,
        verbose_name="active learning",
        help_text="Leave unset to inherit from the study group. Active learning is on by default.",
    )

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS: list[str] = []

    objects = UserManager()

    class Meta:
        db_table = "web_users"
        ordering = ["-date_joined"]

    @property
    def export_id(self) -> str:
        """The identifier this user appears under in exported research data.

        Never the email address. Falls back to the primary key -- a random
        UUID, which identifies the row without identifying the person -- so a
        participant invited by real email still exports de-identified rather
        than leaking an address into the analytic dataset.
        """
        return (self.participant_code or "").strip() or str(self.pk)


class UserConfig(models.Model):
    user = models.OneToOneField(User, models.CASCADE, primary_key=True, related_name="config_row")
    config = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "web_user_configs"


class AuditLog(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    actor = models.ForeignKey(User, models.SET_NULL, null=True, blank=True, related_name="audit_actions")
    action = models.CharField(max_length=64)
    target_type = models.CharField(max_length=32, blank=True, null=True)
    target_id = models.CharField(max_length=255, blank=True, null=True)
    details = models.CharField(max_length=1024, blank=True, null=True)

    class Meta:
        db_table = "web_audit_logs"
        ordering = ["-created_at"]


class ChatMessage(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, models.CASCADE, null=True, blank=True, related_name="chat_messages")
    model_id = models.CharField(max_length=255, db_index=True)
    role = models.CharField(max_length=16)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "web_chat_messages"
        indexes = [
            models.Index(fields=["user", "model_id"], name="ix_web_chat_user_model"),
        ]
        ordering = ["created_at"]


class ParticipantSession(models.Model):
    """Tracks a study participant's progress through the flow.

    Stages advance monotonically:
    consent -> instructions -> task -> post_study -> done.
    ``study_group`` and ``model_id`` are snapshotted when the task begins so an
    admin retargeting a group mid-study cannot move a running participant onto a
    different model.
    """

    STAGE_CONSENT = "consent"
    STAGE_INSTRUCTIONS = "instructions"
    STAGE_TASK = "task"
    STAGE_POST = "post_study"
    STAGE_DONE = "done"

    STAGE_CHOICES = [
        (STAGE_CONSENT, "Consent"),
        (STAGE_INSTRUCTIONS, "Instructions"),
        (STAGE_TASK, "Exploration task"),
        (STAGE_POST, "Post-study questions"),
        (STAGE_DONE, "Complete"),
    ]
    STAGE_ORDER = [
        STAGE_CONSENT,
        STAGE_INSTRUCTIONS,
        STAGE_TASK,
        STAGE_POST,
        STAGE_DONE,
    ]

    user = models.OneToOneField(
        User, models.CASCADE, primary_key=True, related_name="participant_session"
    )
    study_group = models.ForeignKey(
        StudyGroup, models.SET_NULL, null=True, blank=True, related_name="sessions"
    )
    model_id = models.CharField(max_length=255, blank=True)
    # Snapshotted alongside the dashboard so flipping the switch mid-study cannot
    # change the condition a running participant is already in.
    active_learning = models.BooleanField(default=True)
    stage = models.CharField(
        max_length=16, choices=STAGE_CHOICES, default=STAGE_CONSENT, db_index=True
    )

    # Informed consent. The study runs on implied consent -- a participant
    # consents by clicking through -- so the record of it is this timestamp plus
    # the version of the text they were shown. Storing the version means a later
    # revision to the consent document stays attributable per participant
    # instead of being retconned across the whole dataset.
    consented_at = models.DateTimeField(null=True, blank=True)
    consent_version = models.CharField(max_length=64, blank=True)

    # The protocol describes two ways of running a session: independently, or as
    # a live remote one-on-one where the participant thinks aloud and the
    # researcher takes written notes. Which one this was is part of the record.
    moderated = models.BooleanField(
        default=False,
        help_text="Run as a live remote one-on-one session with a researcher present.",
    )

    #: Snapshotted at task start with the rest of the condition, so changing an
    #: override mid-study cannot alter a clock that is already running.
    task_minutes = models.PositiveIntegerField(null=True, blank=True)

    task_started_at = models.DateTimeField(null=True, blank=True)
    #: Projected end, recomputed on every ping. Display only -- the clock is
    #: driven by accumulated presence, not by this timestamp.
    task_ends_at = models.DateTimeField(null=True, blank=True)

    #: Seconds actually spent in the app during the task. The clock pauses when
    #: the participant is gone: a request that arrives more than
    #: PRESENCE_MAX_GAP_S after the previous one contributes nothing, so signing
    #: out, closing the laptop or losing the connection costs no exploration
    #: time. Wall-clock elapsed remains derivable from the two timestamps, and
    #: the difference between them is time away.
    task_elapsed_s = models.PositiveIntegerField(default=0)
    #: When this participant was last seen making a request during the task.
    last_seen_at = models.DateTimeField(null=True, blank=True)
    task_completed_at = models.DateTimeField(null=True, blank=True)
    finished_early = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "web_participant_sessions"
        ordering = ["study_group", "user__email"]

    def __str__(self) -> str:
        return f"{self.user.email} ({self.stage})"

    #: A request arriving within this many seconds of the previous one counts as
    #: continuous presence. Longer, and the participant is treated as away and
    #: the gap is not charged. Comfortably above the timer widget's 30s poll, so
    #: one dropped request does not read as an absence.
    PRESENCE_MAX_GAP_S = 90

    @property
    def task_total_seconds(self) -> int:
        from django.conf import settings

        minutes = self.task_minutes or getattr(settings, "STUDY_TASK_MINUTES", 30)
        return int(minutes) * 60

    @property
    def active_seconds(self) -> int:
        """Time spent in the app, including the stretch since the last ping."""
        from django.utils import timezone

        total = int(self.task_elapsed_s or 0)
        if self.last_seen_at and self.stage == self.STAGE_TASK:
            gap = (timezone.now() - self.last_seen_at).total_seconds()
            if 0 < gap <= self.PRESENCE_MAX_GAP_S:
                total += int(gap)
        return total

    @property
    def seconds_remaining(self) -> int:
        if not self.task_started_at:
            return 0
        return max(0, self.task_total_seconds - self.active_seconds)

    @property
    def expired(self) -> bool:
        return bool(self.task_started_at) and self.seconds_remaining <= 0

    @property
    def task_duration_s(self) -> int | None:
        """Time actually spent exploring. Compare with task_wallclock_s."""
        if not self.task_started_at:
            return None
        return int(self.task_elapsed_s or 0)

    @property
    def task_wallclock_s(self) -> int | None:
        """Start to finish including any time away, for reference."""
        if not (self.task_started_at and self.task_completed_at):
            return None
        return int((self.task_completed_at - self.task_started_at).total_seconds())

    @property
    def seconds_away(self) -> int | None:
        """Wall-clock minus presence: how long they were gone mid-task."""
        wall = self.task_wallclock_s
        if wall is None:
            return None
        return max(0, wall - int(self.task_elapsed_s or 0))

    def advance_to(self, stage: str) -> bool:
        """Move forward to *stage*. Never moves backwards. Returns True if moved."""
        if stage not in self.STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage!r}")
        if self.STAGE_ORDER.index(stage) <= self.STAGE_ORDER.index(self.stage):
            return False
        self.stage = stage
        return True


class StudyResponse(models.Model):
    """A participant's post-study submission. One per participant."""

    user = models.OneToOneField(
        User, models.CASCADE, primary_key=True, related_name="study_response"
    )
    study_group = models.ForeignKey(
        StudyGroup, models.SET_NULL, null=True, blank=True, related_name="responses"
    )
    model_id = models.CharField(max_length=255, blank=True)

    #: Which version of the instrument this was scored against. Without it a
    #: mid-study wording change would make old and new scores silently
    #: incomparable. See web.instruments.
    instrument_version = models.CharField(max_length=64, blank=True)

    tf_answers = models.JSONField(default=dict, blank=True)
    tf_score = models.IntegerField(default=0)
    #: Statements given a substantive answer. Read with ``tf_score``: it
    #: separates "got it wrong" from "chose not to answer", which the consent
    #: explicitly permits.
    tf_answered = models.IntegerField(default=0)
    open_answers = models.JSONField(default=dict, blank=True)
    payload = models.JSONField(default=dict, blank=True)

    task_duration_s = models.PositiveIntegerField(null=True, blank=True)
    form_duration_ms = models.PositiveIntegerField(null=True, blank=True)
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "web_study_responses"
        ordering = ["-submitted_at"]

    def __str__(self) -> str:
        return f"{self.user.email} ({self.tf_score} correct)"


class TopicRename(models.Model):
    """A participant's private label for one topic of one model.

    Renames used to be proxied upstream, but that endpoint does not exist and the
    view silently swallowed the failure. Storing them here also keeps one
    participant's renames from being visible to the rest of their study group.
    """

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, models.CASCADE, related_name="topic_renames")
    model_id = models.CharField(max_length=255)
    topic_id = models.CharField(max_length=64)
    label = models.CharField(max_length=255)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "web_topic_renames"
        ordering = ["user", "model_id", "topic_id"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "model_id", "topic_id"], name="uq_topic_rename"
            ),
        ]
        indexes = [
            models.Index(fields=["user", "model_id"], name="ix_topic_rename_user_model"),
        ]

    def __str__(self) -> str:
        return f"{self.model_id}/{self.topic_id} -> {self.label}"


class ResearcherNote(models.Model):
    """A note taken by a researcher during a moderated session.

    The approved protocol names moderated-session notes as research records, so
    they belong in the dataset keyed to the participant's study code rather than
    in someone's local file where they cannot be linked to anything.

    ``author`` is the staff member who wrote it, kept separate from ``user``
    (the participant the note is about) -- that distinction matters when more
    than one researcher runs sessions.
    """

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        User,
        models.CASCADE,
        related_name="researcher_notes",
        verbose_name="participant",
    )
    author = models.ForeignKey(
        User,
        models.SET_NULL,
        null=True,
        blank=True,
        related_name="authored_notes",
        verbose_name="written by",
    )
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "web_researcher_notes"
        ordering = ["user", "created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"], name="ix_note_user_time"),
        ]

    def __str__(self) -> str:
        return f"{self.user.export_id}: {self.text[:60]}"


class DocumentTheme(models.Model):
    """A user's own theme for one document of one model.

    The model assigns every document a dominant topic. Sometimes it is wrong,
    and a person reading the document knows better. This records that
    correction without touching the model: it is an overlay, applied on read,
    private to whoever made it.

    Kept separate from TopicRename, which renames a topic for every document in
    it. This moves one document to a different theme.
    """

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, models.CASCADE, related_name="document_themes")
    model_id = models.CharField(max_length=255)
    doc_id = models.CharField(max_length=255)
    theme = models.CharField(max_length=255)
    #: The theme the model had assigned, kept so a correction can be read as a
    #: disagreement rather than just a value.
    previous_theme = models.CharField(max_length=255, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "web_document_themes"
        ordering = ["user", "model_id", "doc_id"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "model_id", "doc_id"], name="uq_document_theme"
            ),
        ]
        indexes = [
            models.Index(fields=["user", "model_id"], name="ix_doctheme_user_model"),
        ]

    def __str__(self) -> str:
        return f"{self.doc_id} -> {self.theme}"


class InteractionEventModality:
    """Modality choices shared by InteractionEvent and LabelEvent.

    Declared once so the two tables cannot drift apart -- they are joined on
    modality during analysis.
    """

    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    OTHER = "other"
    CHOICES = [(MOUSE, "Mouse"), (KEYBOARD, "Keyboard"), (OTHER, "Other")]


class LabelEvent(models.Model):
    """One annotation action by a participant, with how it was arrived at.

    **This is an event log, not state.** The backend remains the single source
    of truth for what the labels *are*: the UI still writes every label through
    to ``/active-learning/{model}/label``, the model retrains there, and the
    dashboard reads from it. Nothing here is ever read back as the current
    label of a document.

    What the backend cannot know is the study context, because the payload it
    receives is only ``{doc_id, label}``: which participant, in which condition,
    whether the label was a suggestion accepted or typed from scratch, which
    rank, how long they read first, keyboard or mouse, and where it fell in
    their session. That is behavioural data about a person rather than
    annotation state about a corpus, and it belongs on this side.

    ``doc_id`` and ``label`` are duplicated here deliberately, so provenance can
    be joined to an outcome and inter-annotator agreement computed without
    depending on the backend being reachable at analysis time.

    It is deliberately denormalised: each row carries what the model was
    proposing *at the moment of the action*, because that is unrecoverable
    afterwards once the model has been retrained on the new labels.

    Serves four questions:

    * are the suggestions useful -- ``provenance``, ``suggestion_rank``,
      ``agreed_with_model``
    * are the labels reliable -- ``label`` per ``doc_id`` per participant, so
      agreement between people on the same dashboard is computable
    * does active learning converge -- ``sequence_index`` against
      ``model_confidence`` over a session
    * is annotation efficient -- ``dwell_ms`` and ``modality``
    """

    #: Where the applied label came from.
    FROM_SUGGESTION = "suggestion"   # clicked or keyed a suggested chip
    FROM_TYPED = "typed"             # free text, including datalist picks
    FROM_UNKNOWN = "unknown"         # client sent no provenance
    PROVENANCE_CHOICES = [
        (FROM_SUGGESTION, "Accepted a suggestion"),
        (FROM_TYPED, "Typed their own"),
        (FROM_UNKNOWN, "Unknown"),
    ]

    APPLIED = "applied"
    REMOVED = "removed"
    ACTION_CHOICES = [(APPLIED, "Applied"), (REMOVED, "Removed")]

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, models.CASCADE, related_name="label_events")
    study_group = models.ForeignKey(
        StudyGroup, models.SET_NULL, null=True, blank=True, related_name="label_events"
    )
    model_id = models.CharField(max_length=255, db_index=True)
    doc_id = models.CharField(max_length=255, db_index=True)

    action = models.CharField(max_length=8, choices=ACTION_CHOICES, default=APPLIED)
    label = models.CharField(max_length=255, blank=True)

    provenance = models.CharField(
        max_length=16, choices=PROVENANCE_CHOICES, default=FROM_UNKNOWN
    )
    #: 1-based position of the accepted suggestion; null when typed.
    suggestion_rank = models.PositiveSmallIntegerField(null=True, blank=True)
    #: The suggestions on offer when the action happened, in order.
    suggestions_shown = models.JSONField(default=list, blank=True)

    #: The model's own label for this document, and its confidence in it.
    #: Document-level: the backend does not return per-suggestion scores.
    model_label = models.CharField(max_length=255, blank=True)
    model_confidence = models.FloatField(null=True, blank=True)
    #: Whether the applied label matched the model's own label.
    agreed_with_model = models.BooleanField(null=True, blank=True)

    #: Applied to several documents at once from the bulk bar.
    bulk = models.BooleanField(default=False)
    #: Changed a label this participant had already applied to this document.
    is_revision = models.BooleanField(default=False)
    previous_label = models.CharField(max_length=255, blank=True)

    #: Milliseconds between the document being shown and the label applied.
    dwell_ms = models.PositiveIntegerField(null=True, blank=True)
    modality = models.CharField(
        max_length=8, choices=InteractionEventModality.CHOICES, default="other"
    )
    #: 0-based position of this action within the participant's session, so a
    #: convergence curve can be drawn without re-sorting on timestamps.
    sequence_index = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "web_label_events"
        ordering = ["user", "sequence_index", "id"]
        indexes = [
            models.Index(fields=["user", "model_id"], name="ix_levent_user_model"),
            models.Index(fields=["model_id", "doc_id"], name="ix_levent_model_doc"),
        ]

    def __str__(self) -> str:
        return f"{self.user_id} {self.action} {self.doc_id} -> {self.label}"[:80]


class InteractionEvent(models.Model):
    """One participant interaction, tagged with how it was made.

    The point of this table is the keyboard-vs-mouse comparison: every logged
    action carries a ``modality`` so the two can be counted against each other
    per participant and per condition.

    Deliberately **not** a keylogger. Printable characters are recorded as the
    single token ``"char"`` -- the fact that typing happened, never what was
    typed. Named keys (Enter, ArrowRight, Escape) and modifier chords are kept
    verbatim, because those are the shortcuts worth measuring.
    """

    MOUSE = InteractionEventModality.MOUSE
    KEYBOARD = InteractionEventModality.KEYBOARD
    OTHER = InteractionEventModality.OTHER
    MODALITY_CHOICES = InteractionEventModality.CHOICES

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, models.CASCADE, related_name="interaction_events")
    study_group = models.ForeignKey(
        StudyGroup, models.SET_NULL, null=True, blank=True, related_name="interaction_events"
    )
    model_id = models.CharField(max_length=255, blank=True)
    stage = models.CharField(max_length=16, blank=True)
    page = models.CharField(max_length=64, blank=True)

    action = models.CharField(max_length=32)
    modality = models.CharField(max_length=8, choices=MODALITY_CHOICES, default=OTHER)
    target = models.CharField(max_length=255, blank=True)
    detail = models.JSONField(default=dict, blank=True)

    #: Milliseconds since the page loaded, as reported by the client.
    client_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "web_interaction_events"
        ordering = ["user", "created_at", "id"]
        indexes = [
            models.Index(fields=["user", "created_at"], name="ix_ievent_user_time"),
            models.Index(fields=["model_id", "modality"], name="ix_ievent_model_mode"),
        ]

    def __str__(self) -> str:
        return f"{self.user_id} {self.modality}:{self.action} {self.target}"[:80]
