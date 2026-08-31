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

    task_started_at = models.DateTimeField(null=True, blank=True)
    task_ends_at = models.DateTimeField(null=True, blank=True)
    task_completed_at = models.DateTimeField(null=True, blank=True)
    finished_early = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "web_participant_sessions"
        ordering = ["study_group", "user__email"]

    def __str__(self) -> str:
        return f"{self.user.email} ({self.stage})"

    @property
    def seconds_remaining(self) -> int:
        from django.utils import timezone

        if not self.task_ends_at:
            return 0
        return max(0, int((self.task_ends_at - timezone.now()).total_seconds()))

    @property
    def expired(self) -> bool:
        from django.utils import timezone

        return bool(self.task_ends_at and timezone.now() >= self.task_ends_at)

    @property
    def task_duration_s(self) -> int | None:
        if not (self.task_started_at and self.task_completed_at):
            return None
        return int((self.task_completed_at - self.task_started_at).total_seconds())

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

    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    OTHER = "other"
    MODALITY_CHOICES = [(MOUSE, "Mouse"), (KEYBOARD, "Keyboard"), (OTHER, "Other")]

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
