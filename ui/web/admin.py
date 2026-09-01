"""Django admin registration for TOVA web models."""

from __future__ import annotations

from django import forms
from django.contrib import admin, messages
from django.contrib.admin.widgets import FilteredSelectMultiple
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.db.models import Count
from django.http import HttpResponseRedirect
from django.db import transaction
from django.shortcuts import render
from django.urls import reverse
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _

from web import study_flow
from web.study_features import FEATURE_ACTIVE_LEARNING, user_has_feature
from web.models import (
    AuditLog,
    ChatMessage,
    InteractionEvent,
    LabelEvent,
    ParticipantSession,
    ResearcherNote,
    StudyGroup,
    StudyResponse,
    TopicRename,
    User,
    UserConfig,
)

# Branding (templates/admin/base_site.html uses these)
admin.site.site_header = _("TOVA")
admin.site.site_title = _("TOVA Admin")
admin.site.index_title = _("TOVA site administration")


#: Login domains that mark an account as a generated study login rather than a
#: real person's mailbox. An address at one of these is itself a study code, so
#: its local part becomes the participant code.
STUDY_LOGIN_DOMAINS = {"study.local", "example.com", "localhost", "invalid"}


def participant_code_for(email: str) -> str | None:
    """Study code for a generated participant login, or None for a real address.

    ``p001@study.local`` -> ``p001``. A real mailbox gets no code, because
    deriving one from it would carry the identifier it is supposed to replace
    into the analytic dataset (the local part of a work address is usually a
    name). Those accounts fall back to ``User.export_id``, which uses the row's
    random UUID instead.
    """
    email = (email or "").strip().lower()
    if "@" not in email:
        return None
    local, _, domain = email.partition("@")
    if domain not in STUDY_LOGIN_DOMAINS:
        return None
    if not local:
        return None

    # Two study domains can yield the same local part -- p001@study.local and
    # p001@example.com both want "p001". The column is unique, so taking it
    # blindly would raise IntegrityError and 500 the create-participants page.
    # Yielding the code instead leaves that account exporting under its UUID,
    # which is still de-identified; a crash mid-cohort would be worse.
    taken = (
        User.objects.filter(participant_code=local).exclude(email=email).exists()
    )
    return None if taken else local


def _iso(value):
    """ISO-8601 for a datetime, empty string for None. Keeps CSV cells uniform."""
    return value.isoformat() if value else ""


#: model_id -> corpus_id, filled in by available_dashboards(). The study varies
#: the model backbone over ONE corpus, so which corpus a dashboard is actually
#: built on is the thing that has to be verified before recruiting.
_CORPUS_BY_MODEL: dict[str, str] = {}


def corpus_for_model(model_id: str) -> str:
    """Corpus a trained model was built on, or "" if unknown."""
    if model_id not in _CORPUS_BY_MODEL:
        available_dashboards()
    return _CORPUS_BY_MODEL.get(model_id, "")


def available_dashboards() -> list[tuple[str, str]]:
    """(model_id, label) for every trained model, for dashboard pickers.

    A dashboard in TOVA is a trained model -- the same list the
    /admin/tova/dashboards console shows. Returns an empty list if the API or
    drafts store is unreachable, so the admin still loads.
    """
    try:
        from tova.api.models.data_schemas import DraftType
        from tova.core import drafts
    except Exception:  # pragma: no cover - import guard for partial installs
        return []

    out: list[tuple[str, str]] = []
    try:
        for draft in drafts.list_drafts(type=DraftType.model):
            meta = draft.metadata or {}
            tr = meta.get("tr_params") or {}
            corpus = meta.get("corpus_id") or tr.get("corpus_id") or ""
            if not corpus:
                # A model with no corpus cannot render a dashboard -- the data
                # call fails outright. Keep those out of the picker so an admin
                # cannot hand a participant a broken page.
                continue
            name = meta.get("name") or tr.get("model_name") or ""
            label = f"{name or draft.id}  ({draft.id} \u00b7 corpus {corpus[:14]})"
            out.append((draft.id, label))
            _CORPUS_BY_MODEL[draft.id] = corpus
    except Exception:
        return []
    out.sort(key=lambda pair: pair[1].lower())
    return out


def _dashboard_choices(current: str = "", *, blank_label: str) -> list[tuple[str, str]]:
    choices = [("", blank_label)] + available_dashboards()
    known = {value for value, _label in choices}
    if current and current not in known:
        # Never let a stale or hand-entered id block saving the form.
        choices.append((current, f"{current}  (not found in the model store)"))
    return choices


class DashboardChoiceMixin:
    """Turns a free-text model id field into a picker of real trained models."""

    dashboard_field = "study_model_id"
    dashboard_blank_label = _("\u2014 no dashboard assigned \u2014")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        field = self.fields.get(self.dashboard_field)
        if field is None:
            return
        current = ""
        if self.instance is not None:
            current = getattr(self.instance, self.dashboard_field, "") or ""
        choices = _dashboard_choices(current, blank_label=self.dashboard_blank_label)
        if len(choices) == 1:
            # Nothing to pick from; leave it as free text rather than a dead select.
            field.help_text = _(
                "No trained models found. Train a model first, or paste an ID."
            )
            return
        self.fields[self.dashboard_field] = forms.ChoiceField(
            choices=choices,
            required=False,
            label=field.label,
            help_text=field.help_text,
        )


class UserAdminForm(DashboardChoiceMixin, forms.ModelForm):
    class Meta:
        model = User
        fields = "__all__"


class StudyGroupAdminForm(DashboardChoiceMixin, forms.ModelForm):
    """A study group is a condition: one dashboard, one active-learning setting.

    Both live here rather than on the participant, so a condition is defined
    once and everyone in it inherits from the same place.
    """

    dashboard_field = "default_model_id"
    dashboard_blank_label = _("\u2014 no dashboard yet \u2014")

    members = forms.ModelMultipleChoiceField(
        queryset=User.objects.none(),
        required=False,
        widget=FilteredSelectMultiple(verbose_name=_("users"), is_stacked=False),
        label=_("Members"),
        help_text=_(
            "Move people across to put them in this group. Removing someone here "
            "clears their group but leaves any dashboard assigned to them directly."
        ),
    )

    class Meta:
        model = StudyGroup
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dashboard = self.fields.get("default_model_id")
        if dashboard is not None:
            dashboard.label = _("Dashboard for this condition")
            dashboard.help_text = _(
                "The trained model everyone in this condition works with. "
                "Participants inherit it, so this is the only place it is set."
            )
        al = self.fields.get("enable_active_learning")
        if al is not None:
            al.label = _("Active learning in this condition")
            al.help_text = _(
                "Whether participants in this condition get the active-learning "
                "tab. Set it here, not when creating participants."
            )
        # Staff accounts are never study participants, so keep them out of the picker.
        self.fields["members"].queryset = User.objects.filter(
            is_staff=False, is_superuser=False
        ).order_by("email")
        if self.instance.pk:
            self.fields["members"].initial = self.instance.users.values_list("pk", flat=True)


class ConditionChoiceField(forms.ModelMultipleChoiceField):
    """Study groups, labelled with the condition they actually represent."""

    def label_from_instance(self, obj):
        dashboard = (obj.default_model_id or "").strip()
        if dashboard:
            name = dict(available_dashboards()).get(dashboard, dashboard)
            where = name.split("  (")[0]
        else:
            where = "no dashboard assigned"
        al = "active learning ON" if obj.enable_active_learning else "active learning OFF"
        return f"{obj.name} \u2014 {where}, {al}"


class CreateParticipantsForm(forms.Form):
    """Bulk-create participants and deal them into conditions you have defined."""

    count = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=500,
        label=_("How many to generate"),
        help_text=_("Leave blank if you are pasting email addresses instead."),
    )
    emails = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 5, "cols": 50}),
        label=_("Or paste email addresses"),
        help_text=_("One per line, or comma separated. Use this for named participants."),
    )
    prefix = forms.CharField(initial="p", max_length=20, label=_("Login prefix"))
    domain = forms.CharField(initial="study.local", max_length=100, label=_("Login domain"))
    start = forms.IntegerField(
        initial=1,
        min_value=1,
        label=_("Start numbering at"),
        help_text=_("Raise this for a second cohort so logins do not collide."),
    )
    groups = ConditionChoiceField(
        queryset=StudyGroup.objects.none(),
        widget=forms.CheckboxSelectMultiple,
        label=_("Deal them into"),
        help_text=_(
            "Each participant joins exactly ONE condition. Tick several to divide "
            "the cohort evenly between them, round-robin. A condition carries its "
            "own dashboard and its own active-learning setting \u2014 edit those on "
            "the condition itself, not here."
        ),
    )
    skip_existing = forms.BooleanField(
        required=False,
        initial=True,
        label=_("Leave existing accounts untouched"),
        help_text=_("Unticked, an existing account is reassigned and its password reset."),
    )

    #: Set during clean() when real email addresses were pasted. Surfaced on the
    #: form rather than raised, because inviting named people is legitimate.
    uncoded_warning = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conditions = StudyGroup.objects.order_by("name")
        self.fields["groups"].queryset = conditions
        self.fields["groups"].initial = list(conditions.values_list("pk", flat=True))
        self.conditions_available = conditions.exists()

    def clean(self):
        cleaned = super().clean()

        # A condition with no dashboard leaves its participants unable to start,
        # and they would not find out until they signed in.
        undashed = [
            g.name for g in cleaned.get("groups") or []
            if not (g.default_model_id or "").strip()
        ]
        if undashed:
            raise forms.ValidationError(
                _("These conditions have no dashboard, so participants in them "
                  "could not start: %s. Assign one on the condition first.")
                % ", ".join(undashed)
            )

        count = cleaned.get("count")
        emails_raw = (cleaned.get("emails") or "").strip()

        if bool(count) == bool(emails_raw):
            raise forms.ValidationError(
                _("Give either a number to generate or a list of email addresses, not both.")
            )

        if emails_raw:
            parts = [
                chunk.strip().lower()
                for line in emails_raw.splitlines()
                for chunk in line.split(",")
                if chunk.strip()
            ]
            seen, emails = set(), []
            for item in parts:
                if item not in seen:
                    seen.add(item)
                    emails.append(item)
            invalid = [e for e in emails if "@" not in e]
            if invalid:
                raise forms.ValidationError(
                    _("These do not look like email addresses: %s") % ", ".join(invalid[:5])
                )

            # Real mailboxes get no study code, so their data exports under the
            # row's UUID instead. That is still de-identified, but it loses the
            # readable p001/p002 labelling -- worth saying out loud rather than
            # discovering at analysis time.
            uncoded = [e for e in emails if not participant_code_for(e)]
            if uncoded:
                self.uncoded_warning = _(
                    "%(n)d of these are real email addresses, so they get no study "
                    "code and will export under an internal ID instead of p001-style "
                    "labels. Generated logins are preferable for analysis."
                ) % {"n": len(uncoded)}
        else:
            start = cleaned.get("start") or 1
            prefix = cleaned.get("prefix") or "p"
            domain = cleaned.get("domain") or "study.local"
            emails = [f"{prefix}{n:03d}@{domain}" for n in range(start, start + count)]

        cleaned["resolved_emails"] = emails
        return cleaned


class AssignDashboardForm(forms.Form):
    """Intermediate page for the bulk 'assign dashboard' action."""

    ACTIVE_LEARNING_CHOICES = [
        ("keep", _("Leave active learning as it is")),
        ("on", _("Turn active learning ON for these users")),
        ("off", _("Turn active learning OFF for these users")),
    ]

    study_model_id = forms.ChoiceField(
        required=False,
        label=_("Dashboard"),
        help_text=_("The trained model these users will work with."),
    )
    active_learning = forms.ChoiceField(
        choices=ACTIVE_LEARNING_CHOICES, initial="keep", label=_("Active learning")
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["study_model_id"].choices = _dashboard_choices(
            "", blank_label=_("\u2014 clear the assignment \u2014")
        )


@admin.register(StudyGroup)
class StudyGroupAdmin(admin.ModelAdmin):
    form = StudyGroupAdminForm
    list_display = ("name", "dashboard_name", "enable_active_learning", "user_count")
    list_filter = ("enable_active_learning",)
    search_fields = ("name", "slug", "default_model_id", "description")
    prepopulated_fields = {"slug": ("name",)}
    readonly_fields = ("member_table",)
    fieldsets = (
        (
            _("The condition"),
            {
                "fields": ("name", "slug", "description", "default_model_id"),
                "description": _(
                    "A condition is one dashboard plus one active-learning setting. "
                    "Everyone assigned here inherits both, so changing them moves the "
                    "whole arm at once. Give it a name you will recognise in the data "
                    "\u2014 it is what appears in the exports."
                ),
            },
        ),
        (
            _("Active learning"),
            {
                "fields": ("enable_active_learning",),
                "description": _(
                    "Define two conditions on the same dashboard, one with this on and "
                    "one with it off, to compare the tool with and without it."
                ),
            },
        ),
        (
            _("Members"),
            {"fields": ("members", "member_table")},
        ),
    )

    @admin.display(description=_("Dashboard"), ordering="default_model_id")
    def dashboard_name(self, obj):
        model_id = (obj.default_model_id or "").strip()
        if not model_id:
            return format_html('<span style="color:#b7791f">no dashboard yet</span>')
        name = dict(available_dashboards()).get(model_id, model_id).split("  (")[0]
        return format_html('<span title="{}">{}</span>', model_id, name)

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_n_users=Count("users"))

    @admin.display(description=_("Users"), ordering="_n_users")
    def user_count(self, obj):
        n = getattr(obj, "_n_users", None)
        if n is None:
            n = obj.users.count()
        url = reverse("admin:web_user_changelist") + f"?study_group__id__exact={obj.pk}"
        return format_html('<a href="{}">{} user{}</a>', url, n, "" if n == 1 else "s")

    @admin.display(description=_("Who is in this group"))
    def member_table(self, obj):
        """Members with the dashboard each will actually land on."""
        if not obj.pk:
            return _("Save the group first, then add members.")

        members = obj.users.all().order_by("email")
        if not members:
            return _("No members yet. Use the picker above to add people.")

        labels = dict(available_dashboards())
        rows = []
        for user in members:
            own = (user.study_model_id or "").strip()
            effective = own or (obj.default_model_id or "").strip()
            name = labels.get(effective, effective).split("  (")[0] or "\u2014"
            source = _("own") if own else (_("from group") if effective else _("none"))
            rows.append(
                format_html(
                    "<tr><td style='padding:2px 14px 2px 0'>"
                    "<a href='{}'>{}</a></td>"
                    "<td style='padding:2px 14px 2px 0'>{}</td>"
                    "<td style='padding:2px 14px 2px 0;color:#666'>{}</td>"
                    "<td style='padding:2px 0'>{}</td></tr>",
                    reverse("admin:web_user_change", args=[user.pk]),
                    user.email,
                    name,
                    source,
                    _("AL on") if user_has_feature(user, FEATURE_ACTIVE_LEARNING) else _("AL off"),
                )
            )
        return format_html(
            "<table><thead><tr>"
            "<th style='text-align:left;padding-right:14px'>{}</th>"
            "<th style='text-align:left;padding-right:14px'>{}</th>"
            "<th style='text-align:left;padding-right:14px'>{}</th>"
            "<th style='text-align:left'>{}</th>"
            "</tr></thead><tbody>{}</tbody></table>",
            _("Participant"), _("Dashboard"), _("Source"), _("Active learning"),
            format_html("".join(rows)),
        )

    def save_related(self, request, form, formsets, change):
        super().save_related(request, form, formsets, change)
        if "members" not in form.cleaned_data:
            return
        group = form.instance
        chosen = form.cleaned_data["members"]
        chosen_ids = [u.pk for u in chosen]

        removed = User.objects.filter(study_group=group).exclude(pk__in=chosen_ids)
        n_removed = removed.update(study_group=None)
        n_added = (
            User.objects.filter(pk__in=chosen_ids)
            .exclude(study_group=group)
            .update(study_group=group)
        )
        if n_added or n_removed:
            self.message_user(
                request,
                _("Group membership updated: %(a)d added, %(r)d removed.")
                % {"a": n_added, "r": n_removed},
                messages.INFO,
            )


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    form = UserAdminForm
    ordering = ("email",)
    list_display = (
        "email",
        "name",
        "dashboard",
        "active_learning",
        "study_group",
        "is_staff",
        "is_admin",
        "is_active",
    )
    list_filter = (
        "study_model_id",
        "enable_active_learning",
        "study_group",
        "is_staff",
        "is_superuser",
        "is_active",
        "is_admin",
        "auth_source",
    )
    search_fields = ("email", "name", "study_model_id")
    readonly_fields = ("date_joined", "last_login")
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        (_("Personal info"), {"fields": ("name", "auth_source")}),
        (
            _("Study assignment"),
            {
                "fields": ("study_model_id", "enable_active_learning", "study_group"),
                "description": _(
                    "Assigning a dashboard makes this user a study participant and "
                    "sends them into the study flow at next login. Users sharing a "
                    "dashboard form one condition."
                ),
            },
        ),
        (
            _("Permissions"),
            {"fields": ("is_active", "is_staff", "is_superuser", "is_admin", "groups", "user_permissions")},
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("email", "name", "password1", "password2"),
            },
        ),
        (
            _("Study assignment"),
            {
                "classes": ("wide",),
                "fields": ("study_model_id", "enable_active_learning"),
                "description": _(
                    "Optional. Assigning a dashboard here makes this user a study "
                    "participant straight away, so you do not have to save and "
                    "reopen the record."
                ),
            },
        ),
    )

    def get_form(self, request, obj=None, **kwargs):
        """Use the dashboard picker on the add form too, not just on edit."""
        form = super().get_form(request, obj, **kwargs)
        if obj is None and "study_model_id" in form.base_fields:
            form.base_fields["study_model_id"] = forms.ChoiceField(
                choices=_dashboard_choices("", blank_label=_("\u2014 no dashboard assigned \u2014")),
                required=False,
                label=_("Assigned dashboard"),
                help_text=_("The trained model backing this participant's dashboard."),
            )
        return form


    # ------------------------------------------------------------------
    # Bulk participant creation
    # ------------------------------------------------------------------

    change_list_template = "admin/web/user/change_list.html"

    def get_urls(self):
        from django.urls import path

        custom = [
            path(
                "create-participants/",
                self.admin_site.admin_view(self.create_participants_view),
                name="web_user_create_participants",
            ),
        ]
        return custom + super().get_urls()

    def create_participants_view(self, request):
        import secrets

        context = {
            **self.admin_site.each_context(request),
            "title": _("Create participants"),
            "opts": self.model._meta,
        }

        if request.method != "POST":
            form = CreateParticipantsForm()
            context.update({"form": form, "conditions_available": form.conditions_available})
            return render(request, "admin/create_participants.html", context)

        form = CreateParticipantsForm(request.POST)
        if not form.is_valid():
            context.update({"form": form, "conditions_available": form.conditions_available})
            return render(request, "admin/create_participants.html", context)

        emails = form.cleaned_data["resolved_emails"]
        conditions = list(form.cleaned_data["groups"])
        skip_existing = form.cleaned_data["skip_existing"]
        labels = dict(available_dashboards())

        rows = []
        with transaction.atomic():
            for index, email in enumerate(emails):
                group = conditions[index % len(conditions)]
                model_id = (group.default_model_id or "").strip()
                user = User.objects.filter(email=email).first()

                if user is not None and skip_existing:
                    rows.append(
                        {
                            "email": email,
                            "password": "",
                            "dashboard": user.study_model_id or "",
                            "dashboard_label": labels.get(user.study_model_id, ""),
                            "condition": user.study_group.name if user.study_group_id else "",
                            "status": "skipped",
                        }
                    )
                    continue

                password = secrets.token_urlsafe(9)
                created = user is None
                if created:
                    user = User(email=email, name=email, auth_source="local")

                # The condition is the single source of truth for both halves:
                # the participant inherits the dashboard and the active-learning
                # setting from their group rather than carrying copies that could
                # drift out of step with it.
                user.study_group = group
                user.study_model_id = ""
                user.enable_active_learning = None
                user.is_active = True
                user.participant_code = participant_code_for(email)
                user.set_password(password)
                user.save()

                # Drop a run that has not started so it cannot hold a stale
                # dashboard; runs already underway are left alone.
                ParticipantSession.objects.filter(
                    user=user, task_started_at__isnull=True
                ).delete()

                rows.append(
                    {
                        "email": email,
                        "password": password,
                        "dashboard": model_id,
                        "dashboard_label": labels.get(model_id, model_id).split("  (")[0],
                        "condition": group.name,
                        "status": "created" if created else "updated",
                    }
                )

        counts = {}
        for row in rows:
            if row["status"] != "skipped":
                key = row["condition"] or row["dashboard_label"]
                counts[key] = counts.get(key, 0) + 1

        context.update(
            {
                "rows": rows,
                "created": sum(1 for r in rows if r["status"] == "created"),
                "updated": sum(1 for r in rows if r["status"] == "updated"),
                "skipped": sum(1 for r in rows if r["status"] == "skipped"),
                "per_dashboard": sorted(counts.items()),
            }
        )
        return render(request, "admin/create_participants_result.html", context)

    @admin.display(description=_("Dashboard"), ordering="study_model_id")
    def dashboard(self, obj):
        """Effective dashboard, linked to everyone else sharing it."""
        labels = dict(available_dashboards())
        model_id = (obj.study_model_id or "").strip()
        inherited = False
        if not model_id and obj.study_group:
            model_id = (obj.study_group.default_model_id or "").strip()
            inherited = bool(model_id)
        if not model_id:
            return "\u2014"

        name = labels.get(model_id, model_id).split("  (")[0]
        if inherited:
            # Inherited, so a study_model_id filter would not find them.
            return format_html("{} <span style='color:#666'>(from group)</span>", name)
        url = reverse("admin:web_user_changelist") + f"?study_model_id={model_id}"
        return format_html('<a href="{}" title="{}">{}</a>', url, model_id, name)

    @admin.display(description=_("Active learning"), boolean=True)
    def active_learning(self, obj):
        return user_has_feature(obj, FEATURE_ACTIVE_LEARNING)

    @admin.action(description=_("Assign dashboard to selected users"))
    def assign_dashboard(self, request, queryset):
        """One action with a picker, rather than one action per group.

        The previous implementation built an action per StudyGroup inside
        get_actions(), which ran a query on every changelist render and grew
        without bound as conditions were added.
        """
        if "apply" in request.POST:
            form = AssignDashboardForm(request.POST)
            if form.is_valid():
                model_id = form.cleaned_data["study_model_id"]
                al = form.cleaned_data["active_learning"]

                updates = {"study_model_id": model_id}
                if al == "on":
                    updates["enable_active_learning"] = True
                elif al == "off":
                    updates["enable_active_learning"] = False

                count = queryset.update(**updates)

                # Sessions that have not started yet should pick up the change;
                # ones already underway keep the condition they began in.
                ParticipantSession.objects.filter(
                    user__in=queryset, task_started_at__isnull=True
                ).delete()

                label = dict(available_dashboards()).get(model_id, model_id) or _("(none)")
                self.message_user(
                    request,
                    _("Assigned %(n)d user(s) to dashboard %(d)s.")
                    % {"n": count, "d": label},
                    messages.SUCCESS,
                )
                return HttpResponseRedirect(request.get_full_path())
        else:
            form = AssignDashboardForm()

        return render(
            request,
            "admin/assign_dashboard.html",
            {
                **self.admin_site.each_context(request),
                "title": _("Assign dashboard"),
                "form": form,
                "users": queryset,
                "opts": self.model._meta,
                "action_checkbox_name": admin.helpers.ACTION_CHECKBOX_NAME,
            },
        )

    actions = ["assign_dashboard"]


@admin.register(UserConfig)
class UserConfigAdmin(admin.ModelAdmin):
    list_display = ("user", "updated_at")
    search_fields = ("user__email",)
    readonly_fields = ("updated_at",)


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "actor", "action", "target_type", "target_id")
    list_filter = ("action", "target_type")
    search_fields = ("details", "target_id", "actor__email")
    readonly_fields = ("created_at", "actor", "action", "target_type", "target_id", "details")
    date_hierarchy = "created_at"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return request.user.is_active and request.user.is_staff

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user", "model_id", "role")
    list_filter = ("role", "model_id")
    search_fields = ("content", "user__email", "model_id")
    readonly_fields = ("created_at", "user", "model_id", "role", "content")
    date_hierarchy = "created_at"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return request.user.is_active and request.user.is_staff

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser


class _ReadOnlyStudyAdmin(admin.ModelAdmin):
    """Study data is collected, not edited. Superusers may still delete."""

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        return request.user.is_active and request.user.is_staff

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser


@admin.register(ParticipantSession)
class ParticipantSessionAdmin(_ReadOnlyStudyAdmin):
    list_display = (
        "user",
        "study_group",
        "stage",
        "model_id",
        "moderated",
        "task_started_at",
        "task_completed_at",
        "active_time",
        "finished_early",
    )
    list_filter = ("stage", "study_group", "finished_early", "moderated")
    search_fields = ("user__email", "model_id")
    readonly_fields = (
        "user",
        "study_group",
        "model_id",
        "stage",
        "moderated",
        "task_started_at",
        "task_ends_at",
        "task_completed_at",
        "finished_early",
        "created_at",
        "completed_at",
    )
    date_hierarchy = "created_at"

    @admin.display(description=_("Active / wall-clock"))
    def active_time(self, obj):
        """Time in the app versus start-to-finish. A gap means they were away."""
        if not obj.task_started_at:
            return "\u2014"
        wall = obj.task_wallclock_s
        active = int(obj.task_elapsed_s or 0)
        if wall is None:
            return f"{active}s"
        return f"{active}s / {wall}s"

    @admin.action(description="Reset participant (clears their run and response)")
    def reset_participants(self, request, queryset):
        users = list(queryset.values_list("user_id", flat=True))
        StudyResponse.objects.filter(user_id__in=users).delete()
        TopicRename.objects.filter(user_id__in=users).delete()
        count = queryset.delete()[0]
        self.message_user(request, f"Reset {count} participant session(s).")

    @admin.action(description=_("Mark as a moderated session"))
    def mark_moderated(self, request, queryset):
        n = queryset.update(moderated=True)
        self.message_user(request, _("Marked %(n)d session(s) as moderated.") % {"n": n})

    @admin.action(description=_("Mark as an independent session"))
    def mark_independent(self, request, queryset):
        n = queryset.update(moderated=False)
        self.message_user(request, _("Marked %(n)d session(s) as independent.") % {"n": n})

    actions = ["reset_participants", "mark_moderated", "mark_independent"]

    def has_delete_permission(self, request, obj=None):
        # Needed so the reset action can run; still staff-gated by the changelist.
        return request.user.is_staff


@admin.register(StudyResponse)
class StudyResponseAdmin(_ReadOnlyStudyAdmin):
    list_display = (
        "participant",
        "study_group",
        "tf_score",
        "tf_answered",
        "task_duration_s",
        "submitted_at",
    )
    list_filter = ("study_group", "tf_score", "model_id")
    search_fields = ("user__email", "user__participant_code", "model_id")
    readonly_fields = (
        "user",
        "study_group",
        "model_id",
        "tf_answers",
        "tf_score",
        "tf_answered",
        "open_answers",
        "payload",
        "task_duration_s",
        "form_duration_ms",
        "submitted_at",
    )
    date_hierarchy = "submitted_at"
    change_list_template = "admin/web/studyresponse/change_list.html"
    actions = ["export_selected_csv"]

    @admin.display(description=_("Participant"), ordering="user__participant_code")
    def participant(self, obj):
        return obj.user.export_id if obj.user_id else "—"

    def get_urls(self):
        from django.urls import path

        return [
            path(
                "responses.csv",
                self.admin_site.admin_view(self.responses_csv),
                name="web_studyresponse_responses_csv",
            ),
        ] + super().get_urls()

    # ------------------------------------------------------------------
    # Analysis export
    # ------------------------------------------------------------------

    #: One row per participant. Deliberately flat -- the point is a file that
    #: goes straight into R or pandas without unpacking JSON blobs by hand.
    @property
    def EXPORT_COLUMNS(self):
        cols = [
            "participant", "dashboard", "study_group", "active_learning",
            "task_minutes", "consent_version", "consented_at", "instrument_version",
            "task_started_at", "task_completed_at",
            "task_duration_s", "task_wallclock_s", "seconds_away",
            "finished_early", "moderated",
        ]
        for key in study_flow.TF_KEY:
            cols += [key, f"{key}_correct"]
        cols += ["tf_score", "tf_answered"]
        for key in study_flow.OPEN_QUESTIONS:
            cols += [key, f"{key}_citations"]
        cols += ["form_duration_ms", "submitted_at"]
        return cols

    def _stream_responses(self, queryset, filename):
        """Stream one row per participant, keyed by study code.

        Streamed and iterated the same way as the interaction export so a large
        cohort does not have to fit in memory.
        """
        import csv as _csv

        from django.http import StreamingHttpResponse

        class _Echo:
            def write(self, value):
                return value

        writer = _csv.writer(_Echo())
        columns = self.EXPORT_COLUMNS
        qs = queryset.select_related(
            "user", "study_group", "user__participant_session"
        ).order_by("user__participant_code", "user_id")

        def rows():
            yield writer.writerow(columns)
            for r in qs.iterator(chunk_size=500):
                session = getattr(r.user, "participant_session", None) if r.user_id else None
                answers = r.tf_answers or {}
                open_answers = r.open_answers or {}

                row = [
                    r.user.export_id if r.user_id else "",
                    r.model_id,
                    r.study_group.name if r.study_group_id else "",
                    "" if session is None else int(bool(session.active_learning)),
                    getattr(session, "task_minutes", None) or "",
                    getattr(session, "consent_version", "") or "",
                    _iso(getattr(session, "consented_at", None)),
                    r.instrument_version,
                    _iso(getattr(session, "task_started_at", None)),
                    _iso(getattr(session, "task_completed_at", None)),
                    # Time actually in the app, then start-to-finish, then the
                    # difference -- how long they were away mid-task.
                    r.task_duration_s if r.task_duration_s is not None else "",
                    getattr(session, "task_wallclock_s", None) or "",
                    getattr(session, "seconds_away", None) or "",
                    "" if session is None else int(bool(session.finished_early)),
                    "" if session is None else int(bool(getattr(session, "moderated", False))),
                ]
                for key, correct in study_flow.TF_KEY.items():
                    given = answers.get(key, "")
                    row += [given, int(given == correct)]
                row += [r.tf_score, r.tf_answered]
                for key in study_flow.OPEN_QUESTIONS:
                    entry = open_answers.get(key) or {}
                    row += [
                        entry.get("text", ""),
                        " ".join(entry.get("citations") or []),
                    ]
                row += [
                    r.form_duration_ms if r.form_duration_ms is not None else "",
                    _iso(r.submitted_at),
                ]
                yield writer.writerow(row)

        resp = StreamingHttpResponse(rows(), content_type="text/csv")
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp

    def responses_csv(self, request):
        """Every response. Linked from the changelist."""
        return self._stream_responses(
            StudyResponse.objects.all(), "tova-study-responses.csv"
        )

    @admin.action(description=_("Export selected responses to CSV"))
    def export_selected_csv(self, request, queryset):
        return self._stream_responses(queryset, "tova-study-responses-selection.csv")


@admin.register(ResearcherNote)
class ResearcherNoteAdmin(admin.ModelAdmin):
    """Writable, unlike the other study tables.

    Moderated-session notes are typed by a researcher during the session, so
    this is the one study model that has to be editable in the admin. The
    author defaults to whoever is signed in.
    """

    list_display = ("participant", "author", "preview", "created_at")
    list_filter = ("author", "created_at")
    search_fields = ("text", "user__email", "user__participant_code")
    autocomplete_fields = ("user", "author")
    readonly_fields = ("created_at", "updated_at")
    date_hierarchy = "created_at"

    @admin.display(description=_("Participant"), ordering="user__participant_code")
    def participant(self, obj):
        return obj.user.export_id if obj.user_id else "\u2014"

    @admin.display(description=_("Note"))
    def preview(self, obj):
        text = (obj.text or "").strip().replace("\n", " ")
        return (text[:90] + "\u2026") if len(text) > 90 else text

    def save_model(self, request, obj, form, change):
        if obj.author_id is None:
            obj.author = request.user
        super().save_model(request, obj, form, change)


@admin.register(TopicRename)
class TopicRenameAdmin(_ReadOnlyStudyAdmin):
    list_display = ("user", "model_id", "topic_id", "label", "updated_at")
    list_filter = ("model_id",)
    search_fields = ("user__email", "model_id", "label")
    readonly_fields = ("user", "model_id", "topic_id", "label", "updated_at")


@admin.register(LabelEvent)
class LabelEventAdmin(_ReadOnlyStudyAdmin):
    """The annotation record: what was labelled, and how it was arrived at."""

    list_display = (
        "created_at", "participant", "doc_id", "label", "provenance",
        "suggestion_rank", "agreed_with_model", "model_confidence",
        "dwell_ms", "modality", "sequence_index",
    )
    list_filter = ("action", "provenance", "agreed_with_model", "bulk",
                   "is_revision", "modality", "model_id")
    search_fields = ("user__participant_code", "user__email", "doc_id", "label")
    readonly_fields = tuple(f.name for f in LabelEvent._meta.fields)
    date_hierarchy = "created_at"
    change_list_template = "admin/web/labelevent/change_list.html"
    actions = ["export_selected_csv"]

    @admin.display(description=_("Participant"), ordering="user__participant_code")
    def participant(self, obj):
        return obj.user.export_id if obj.user_id else "\u2014"

    def get_urls(self):
        from django.urls import path

        return [
            path(
                "labels.csv",
                self.admin_site.admin_view(self.labels_csv),
                name="web_labelevent_labels_csv",
            ),
        ] + super().get_urls()

    EXPORT_COLUMNS = [
        "created_at", "participant", "study_group", "dashboard", "sequence_index",
        "doc_id", "action", "label", "provenance", "suggestion_rank",
        "suggestions_shown", "model_label", "model_confidence", "agreed_with_model",
        "bulk", "is_revision", "previous_label", "dwell_ms", "modality",
    ]

    def _stream(self, queryset, filename):
        import csv as _csv
        import json as _json

        from django.http import StreamingHttpResponse

        class _Echo:
            def write(self, value):
                return value

        writer = _csv.writer(_Echo())
        qs = queryset.select_related("user", "study_group").order_by(
            "user__participant_code", "user_id", "sequence_index", "id"
        )

        def rows():
            yield writer.writerow(self.EXPORT_COLUMNS)
            for e in qs.iterator(chunk_size=2000):
                yield writer.writerow([
                    e.created_at.isoformat(),
                    e.user.export_id if e.user_id else "",
                    e.study_group.name if e.study_group_id else "",
                    e.model_id,
                    e.sequence_index,
                    e.doc_id,
                    e.action,
                    e.label,
                    e.provenance,
                    e.suggestion_rank if e.suggestion_rank is not None else "",
                    _json.dumps(e.suggestions_shown, separators=(",", ":")) if e.suggestions_shown else "",
                    e.model_label,
                    e.model_confidence if e.model_confidence is not None else "",
                    "" if e.agreed_with_model is None else int(e.agreed_with_model),
                    int(e.bulk),
                    int(e.is_revision),
                    e.previous_label,
                    e.dwell_ms if e.dwell_ms is not None else "",
                    e.modality,
                ])

        resp = StreamingHttpResponse(rows(), content_type="text/csv")
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp

    def labels_csv(self, request):
        return self._stream(LabelEvent.objects.all(), "tova-label-events.csv")

    @admin.action(description=_("Export selected label events to CSV"))
    def export_selected_csv(self, request, queryset):
        return self._stream(queryset, "tova-label-events-selection.csv")


@admin.register(InteractionEvent)
class InteractionEventAdmin(_ReadOnlyStudyAdmin):
    list_display = ("created_at", "user", "modality", "action", "target", "page", "model_id")
    list_filter = ("modality", "action", "page", "model_id", "stage")
    search_fields = ("user__email", "target", "model_id")
    readonly_fields = (
        "user", "study_group", "model_id", "stage", "page",
        "action", "modality", "target", "detail", "client_ms", "created_at",
    )
    date_hierarchy = "created_at"
    change_list_template = "admin/web/interactionevent/change_list.html"

    actions = ["export_selected_csv"]

    def get_urls(self):
        from django.urls import path

        return [
            path(
                "comparison/",
                self.admin_site.admin_view(self.comparison_view),
                name="web_interactionevent_comparison",
            ),
            path(
                "comparison.csv",
                self.admin_site.admin_view(self.comparison_csv),
                name="web_interactionevent_comparison_csv",
            ),
            path(
                "events.csv",
                self.admin_site.admin_view(self.events_csv),
                name="web_interactionevent_events_csv",
            ),
        ] + super().get_urls()

    # ------------------------------------------------------------------
    # Raw event export
    # ------------------------------------------------------------------

    EXPORT_COLUMNS = [
        "created_at", "participant", "dashboard", "study_group", "stage",
        "page", "action", "modality", "target", "client_ms", "detail",
    ]

    def _stream_events(self, queryset, filename):
        """Stream the queryset as CSV, one row per event.

        Streamed and iterated with .iterator() so a long study does not have to
        fit in memory to be exported.
        """
        import csv as _csv
        import json as _json

        from django.http import StreamingHttpResponse

        class _Echo:
            def write(self, value):
                return value

        writer = _csv.writer(_Echo())
        qs = queryset.select_related("user", "study_group").order_by("user_id", "created_at", "id")

        def rows():
            yield writer.writerow(self.EXPORT_COLUMNS)
            for e in qs.iterator(chunk_size=2000):
                yield writer.writerow([
                    e.created_at.isoformat(),
                    # Study code, never the email address: the consent says data
                    # is "linked only to a study-specific code".
                    e.user.export_id if e.user_id else "",
                    e.model_id,
                    e.study_group.name if e.study_group_id else "",
                    e.stage,
                    e.page,
                    e.action,
                    e.modality,
                    e.target,
                    e.client_ms,
                    _json.dumps(e.detail, separators=(",", ":")) if e.detail else "",
                ])

        resp = StreamingHttpResponse(rows(), content_type="text/csv")
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp

    def events_csv(self, request):
        """Every event. Linked from the changelist."""
        return self._stream_events(
            InteractionEvent.objects.all(), "tova-interaction-events.csv"
        )

    @admin.action(description=_("Export selected events to CSV"))
    def export_selected_csv(self, request, queryset):
        """Export a subset. Filter the changelist, tick 'select all', then run this."""
        return self._stream_events(queryset, "tova-interaction-events-selection.csv")

    # ------------------------------------------------------------------
    # Keyboard vs mouse
    # ------------------------------------------------------------------

    @staticmethod
    def _rollup(rows):
        """Fold (key, modality, n) tuples into per-key totals and shares."""
        out = {}
        for key, modality, n in rows:
            entry = out.setdefault(
                key, {"mouse": 0, "keyboard": 0, "other": 0, "total": 0}
            )
            entry[modality] = entry.get(modality, 0) + n
            entry["total"] += n
        for entry in out.values():
            decided = entry["mouse"] + entry["keyboard"]
            entry["keyboard_share"] = (
                round(100.0 * entry["keyboard"] / decided, 1) if decided else None
            )
        return out

    def _comparison_data(self):
        qs = InteractionEvent.objects.values("model_id", "modality").annotate(n=Count("id"))
        by_dashboard = self._rollup(
            [(r["model_id"] or "(none)", r["modality"], r["n"]) for r in qs]
        )

        # Keyed by study code, not email: this table is research data and it has
        # a CSV export, so an address here would land straight in the analytic
        # dataset. Mirrors User.export_id, computed in SQL rather than per row.
        qs = (
            InteractionEvent.objects.values(
                "user__participant_code", "user_id", "model_id", "modality"
            )
            .annotate(n=Count("id"))
            .order_by("user__participant_code", "user_id")
        )
        rows, user_dashboard = [], {}
        for r in qs:
            code = (r["user__participant_code"] or "").strip() or str(r["user_id"])
            user_dashboard[code] = r["model_id"] or "(none)"
            rows.append((code, r["modality"], r["n"]))
        by_user = self._rollup(rows)

        shortcuts = (
            InteractionEvent.objects.filter(action="shortcut")
            .values("detail")
            .annotate(n=Count("id"))
        )
        shortcut_counts = {}
        for row in shortcuts:
            key = (row["detail"] or {}).get("key") or "?"
            shortcut_counts[key] = shortcut_counts.get(key, 0) + row["n"]

        labels = dict(available_dashboards())
        return {
            "by_dashboard": sorted(
                (
                    {"key": k, "label": labels.get(k, k).split("  (")[0], **v}
                    for k, v in by_dashboard.items()
                ),
                key=lambda d: d["label"].lower(),
            ),
            "by_user": sorted(
                (
                    {"key": k, "dashboard": user_dashboard.get(k, ""), **v}
                    for k, v in by_user.items()
                ),
                key=lambda d: d["key"],
            ),
            "shortcuts": sorted(
                shortcut_counts.items(), key=lambda kv: -kv[1]
            )[:20],
            "total": InteractionEvent.objects.count(),
        }

    def comparison_view(self, request):
        context = {
            **self.admin_site.each_context(request),
            "title": _("Keyboard vs mouse"),
            "opts": self.model._meta,
            **self._comparison_data(),
        }
        return render(request, "admin/interaction_comparison.html", context)

    def comparison_csv(self, request):
        import csv as _csv
        from django.http import StreamingHttpResponse

        data = self._comparison_data()

        class _Echo:
            def write(self, value):
                return value

        writer = _csv.writer(_Echo())

        def rows():
            yield writer.writerow(
                ["scope", "key", "dashboard", "mouse", "keyboard", "other",
                 "total", "keyboard_share_pct"]
            )
            for r in data["by_dashboard"]:
                yield writer.writerow(
                    ["dashboard", r["label"], r["key"], r["mouse"], r["keyboard"],
                     r["other"], r["total"], r["keyboard_share"]]
                )
            for r in data["by_user"]:
                yield writer.writerow(
                    ["participant", r["key"], r["dashboard"], r["mouse"], r["keyboard"],
                     r["other"], r["total"], r["keyboard_share"]]
                )
            for key, n in data["shortcuts"]:
                yield writer.writerow(["shortcut", key, "", "", n, "", n, ""])

        resp = StreamingHttpResponse(rows(), content_type="text/csv")
        resp["Content-Disposition"] = 'attachment; filename="tova-keyboard-vs-mouse.csv"'
        return resp
