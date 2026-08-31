"""Tests for the participant study flow: stage gating, timer, and submission."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from web.models import (
    InteractionEvent,
    LabelEvent,
    ResearcherNote,
    ParticipantSession,
    StudyGroup,
    StudyResponse,
    TopicRename,
    User,
)
from web import study_flow
from web.study_features import FEATURE_ACTIVE_LEARNING, user_has_feature
from web.services import runtime as R


def consent_for(user):
    """Put *user* past the consent gate, the way clicking through would.

    Consent is stage one for every participant, so tests that are about a later
    stage need this to get there. Goes through ``study_flow.record_consent`` so
    the stamp and the stage transition stay identical to production.
    """
    return study_flow.record_consent(study_flow.get_or_create_session(user))


def visible_text(html: str) -> str:
    """Rendered text with tags dropped and whitespace collapsed.

    Lets a test assert on what a participant actually reads, independent of how
    the markup happens to be wrapped in the template source.
    """
    import re

    text = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&mdash;", "—")
        .replace("&amp;", "&")
        .replace("&nbsp;", " ")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


#: The approved consent text, paragraph by paragraph, exactly as it appears in
#: "TOVA CorpusBio - consent document revised.docx". The study runs on implied
#: consent, so this wording *is* the instrument -- if a well-meaning edit ever
#: paraphrases or trims it on the page, that is a protocol deviation, and this
#: is the test that catches it.
APPROVED_CONSENT_PARAGRAPHS = [
    "Your participation in this research is completely voluntary. You may choose not to take "
    "part at all. If you decide to participate in this research, you may stop participating at "
    "any time. If you decide not to participate in this study or if you stop participating at "
    "any time, you will not be penalized or lose any benefits to which you otherwise qualify. "
    "If you decide to stop taking part in the study, if you have questions, concerns, or "
    "complaints, or if you need to report an injury related to the research, please contact the "
    "investigator:",

    "For this study, you will explore approximately 600 synthetic internal emails from a "
    "fictional biotechnology organization using one assigned interface. Some interfaces use "
    "large language models (LLMs) to generate topics or labels. AI-generated information may be "
    "inaccurate or incomplete. During a fixed 30-minute exploration period, you will browse "
    "emails and work with the assigned tool. You will be told the general kinds of questions "
    "that will follow, but not the exact questions. When exploration ends, the tool will close "
    "and you will answer four true/false/not-sure questions and four written questions from "
    "memory. You may also select the email IDs that support your written answers. You will not "
    "be able to reopen the emails during the questionnaire.",

    "You may participate independently or in a live remote one-on-one session. In a moderated "
    "session, you will share only the study-tool window, think aloud, and answer neutral "
    "questions about how you use and learn from the tool. The researcher will take written "
    "notes. No audio, video, screenshots, or screen recording will occur. The full study will "
    "take approximately 30 to 60 minutes. Payment and any bonus terms will be stated in the "
    "approved recruitment posting before you begin.",

    "Research records may include your questionnaire responses; selected supporting email IDs; "
    "themes, topics, or labels displayed or edited during the session; AI interactions where "
    "applicable; application logs such as time, emails opened, queries, retrieval results, and "
    "topic or label edits; and moderated-session notes where applicable. Data will be linked "
    "only to a study-specific code. A recruitment platform may display account information for "
    "scheduling and payment, but it will not be copied into the analytic dataset. Please do not "
    "enter or say identifying information. The fictional emails discuss organizational "
    "misconduct, data manipulation, and gene-therapy safety issues. You may skip a question, "
    "stop screen sharing, or stop participation at any time. There are no direct benefits. "
    "De-identified responses and AI interactions may be included in research publications or "
    "released research datasets.",

    "The research team will use coded responses, selected evidence, themes or labels, "
    "applicable AI interactions, application logs, and moderated-session notes to compare "
    "factual learning, cross-document reasoning, evidence use, and interaction patterns across "
    "the assigned interfaces. Data will be reviewed and anonymized before any public release, "
    "and results will be reported without direct identifiers.",

    "If you click the link below to continue, you confirm that you are at least 18 years old, "
    "are located in and will participate from the United States, and consent to participate in "
    "this study.",
]


def read_template_source(relative_path: str) -> str:
    """Read a template's raw source off disk, wherever the checkout is rooted.

    A few tests assert on the *text* of the telemetry collector rather than on
    rendered output, so they need the file itself. Resolving through the
    configured template dirs keeps them working whether the suite runs from
    ``ui/`` (templates at ``../templates``) or inside the container (templates at
    ``/app/templates``).
    """
    for directory in settings.TEMPLATES[0]["DIRS"]:
        candidate = Path(directory) / relative_path
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"{relative_path} not found in template dirs: "
        + ", ".join(str(d) for d in settings.TEMPLATES[0]["DIRS"])
    )


class StudyFlowTestCase(TestCase):
    def setUp(self):
        self.group = StudyGroup.objects.create(
            name="Condition A",
            slug="condition-a",
            default_model_id="m_test",
            enable_active_learning=True,
        )
        self.participant = User.objects.create_user(
            email="p001@study.local", password="pw", study_group=self.group,
            participant_code="p001",
        )
        self.outsider = User.objects.create_user(email="nobody@example.com", password="pw")
        self.staff = User.objects.create_user(
            email="admin@example.com", password="pw", is_staff=True
        )

    def login_participant(self, consent=True):
        """Sign a participant in, past the consent gate by default.

        Consent is the first stage, so almost every test here would otherwise
        spend its first line clicking through it to reach the behaviour it
        actually cares about. Recorded directly rather than over HTTP so these
        tests do not depend on the consent view to test the timer or the
        questionnaire. Pass ``consent=False`` to sit at the gate itself.
        """
        self.client.force_login(self.participant)
        if consent:
            consent_for(self.participant)


class StageGatingTests(StudyFlowTestCase):
    def test_participant_starts_at_instructions(self):
        self.login_participant()
        resp = self.client.get(reverse("web:dashboard"), {"model_id": "m_test"})
        self.assertRedirects(
            resp, reverse("web:study_instructions"), fetch_redirect_response=False
        )

    def test_admin_workflows_are_blocked(self):
        self.login_participant()
        resp = self.client.get(reverse("web:load_data_page"))
        self.assertEqual(resp.status_code, 302)
        self.assertNotIn("load-data", resp["Location"])

    def test_cannot_skip_to_post_study(self):
        self.login_participant()
        resp = self.client.get(reverse("web:study_post"))
        self.assertRedirects(
            resp, reverse("web:study_instructions"), fetch_redirect_response=False
        )

    def test_json_endpoints_get_409_not_html_redirect(self):
        self.login_participant()
        resp = self.client.post(
            reverse("web:get_dashboard_data"),
            data="{}",
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp["Content-Type"], "application/json")
        self.assertEqual(resp.json()["stage"], ParticipantSession.STAGE_INSTRUCTIONS)

    def test_staff_are_never_gated(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse("web:load_data_page"))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(ParticipantSession.objects.exists())

    def test_non_participants_are_never_gated(self):
        self.client.force_login(self.outsider)
        resp = self.client.get(reverse("web:home"))
        self.assertEqual(resp.status_code, 200)

    def test_always_allowed_routes_pass_at_any_stage(self):
        self.login_participant()
        self.assertEqual(self.client.get(reverse("web:terms")).status_code, 200)
        self.assertEqual(self.client.get(reverse("web:privacy")).status_code, 200)

    def test_active_learning_blocked_when_group_disables_it(self):
        self.group.enable_active_learning = False
        self.group.save()
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        resp = self.client.get(reverse("web:active_learning_two"), {"model_id": "m_test"})
        self.assertEqual(resp.status_code, 302)


class DashboardOnlyGatingTests(TestCase):
    """Participants with a dashboard but no group must be gated too.

    This is how the admin actually creates people: both the "Create
    participants" flow and the "Assign dashboard" bulk action set
    ``study_model_id`` and never a group. The middleware used to bail out
    unless ``study_group_id`` was set, so exactly the participants the research
    team creates were the ones that could walk straight past the flow -- and
    every gating test above builds its participant *with* a group, which is why
    nothing caught it.
    """

    def setUp(self):
        self.participant = User.objects.create_user(
            email="p001@study.local", password="pw", study_model_id="m_solo"
        )
        self.client.force_login(self.participant)
        consent_for(self.participant)

    def test_is_treated_as_a_participant(self):
        from web.study_features import is_study_participant

        self.assertTrue(is_study_participant(self.participant))
        self.assertIsNone(self.participant.study_group_id)

    def test_cannot_skip_straight_to_the_questionnaire(self):
        resp = self.client.get(reverse("web:study_post"))
        self.assertRedirects(
            resp, reverse("web:study_instructions"), fetch_redirect_response=False
        )

    def test_cannot_reach_admin_workflows(self):
        resp = self.client.get(reverse("web:load_data_page"))
        self.assertEqual(resp.status_code, 302)
        self.assertNotIn("load-data", resp["Location"])

    def test_dashboard_is_gated_until_the_task_begins(self):
        resp = self.client.get(reverse("web:dashboard"), {"model_id": "m_solo"})
        self.assertRedirects(
            resp, reverse("web:study_instructions"), fetch_redirect_response=False
        )

    def test_json_endpoints_get_409(self):
        resp = self.client.post(
            reverse("web:get_dashboard_data"), data="{}", content_type="application/json"
        )
        self.assertEqual(resp.status_code, 409)

    def test_the_flow_still_works_end_to_end(self):
        """Gating them must not mean trapping them."""
        self.client.post(reverse("web:study_begin"))
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_TASK)
        self.assertEqual(session.model_id, "m_solo")


class ConsentTests(StudyFlowTestCase):
    """The study runs on implied consent, so the click is the record."""

    def test_participants_land_on_consent_first(self):
        self.login_participant(consent=False)
        resp = self.client.get(reverse("web:study_instructions"))
        self.assertRedirects(
            resp, reverse("web:study_consent"), fetch_redirect_response=False
        )

    def test_nothing_else_is_reachable_before_consenting(self):
        self.login_participant(consent=False)
        for name in ("study_instructions", "study_post", "study_done", "dashboard"):
            with self.subTest(route=name):
                resp = self.client.get(reverse(f"web:{name}"))
                self.assertEqual(resp.status_code, 302, name)
                self.assertIn(reverse("web:study_consent"), resp["Location"], name)

    def test_cannot_begin_the_task_before_consenting(self):
        self.login_participant(consent=False)
        self.client.post(reverse("web:study_begin"))
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_CONSENT)
        self.assertIsNone(session.task_started_at)

    def test_renders_the_approved_text_verbatim(self):
        """Every approved paragraph, word for word. See the constant's comment."""
        self.login_participant(consent=False)
        resp = self.client.get(reverse("web:study_consent"))
        self.assertEqual(resp.status_code, 200)
        rendered = visible_text(resp.content.decode())

        for i, paragraph in enumerate(APPROVED_CONSENT_PARAGRAPHS):
            with self.subTest(paragraph=i):
                self.assertIn(
                    paragraph,
                    rendered,
                    f"approved consent paragraph {i} is missing or was reworded",
                )

    def test_page_carries_the_investigator_and_irb_identifiers(self):
        self.login_participant(consent=False)
        rendered = visible_text(
            self.client.get(reverse("web:study_consent")).content.decode()
        )
        self.assertIn("Jordan Boyd-Graber", rendered)
        self.assertIn("jbg@umiacs.umd.edu", rendered)
        # The telephone number was removed at the research team's request; the
        # email remains as the route to the investigator that the approved text
        # promises.
        self.assertNotIn("405-6766", rendered)
        self.assertIn(study_flow.IRB_PACKAGE_NUMBER, rendered)
        self.assertIn(study_flow.IRB_CONTACT_EMAIL, rendered)

    def test_participants_can_print_a_copy_for_their_records(self):
        """The approved text promises this affordance explicitly."""
        self.login_participant(consent=False)
        body = self.client.get(reverse("web:study_consent")).content.decode()
        self.assertIn("window.print()", body)
        self.assertIn("@media print", body)

    def test_accepting_records_consent_and_advances(self):
        self.login_participant(consent=False)
        resp = self.client.post(reverse("web:study_consent_accept"))
        self.assertRedirects(
            resp, reverse("web:study_instructions"), fetch_redirect_response=False
        )
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_INSTRUCTIONS)
        self.assertIsNotNone(session.consented_at)
        self.assertEqual(session.consent_version, study_flow.CONSENT_VERSION)

    def test_accepting_twice_does_not_restamp(self):
        """An accidental double-submit must not rewrite when consent happened."""
        self.login_participant(consent=False)
        self.client.post(reverse("web:study_consent_accept"))
        first = ParticipantSession.objects.get(pk=self.participant.pk).consented_at

        self.client.post(reverse("web:study_consent_accept"))
        again = ParticipantSession.objects.get(pk=self.participant.pk).consented_at
        self.assertEqual(first, again)

    def test_get_on_the_accept_route_does_not_consent(self):
        self.login_participant(consent=False)
        self.client.get(reverse("web:study_consent_accept"))
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertIsNone(session.consented_at)
        self.assertEqual(session.stage, ParticipantSession.STAGE_CONSENT)

    def test_declining_records_nothing_and_starts_nothing(self):
        self.login_participant(consent=False)
        resp = self.client.get(reverse("web:study_consent_declined"))
        self.assertEqual(resp.status_code, 200)
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertIsNone(session.consented_at)
        self.assertIsNone(session.task_started_at)
        self.assertEqual(session.stage, ParticipantSession.STAGE_CONSENT)

    def test_a_version_bump_does_not_disturb_someone_already_running(self):
        """Their record keeps the version they actually agreed to."""
        self.login_participant(consent=False)
        self.client.post(reverse("web:study_consent_accept"))
        self.client.post(reverse("web:study_begin"))

        stamped = ParticipantSession.objects.get(pk=self.participant.pk).consent_version
        self.assertEqual(stamped, study_flow.CONSENT_VERSION)

        with mock.patch.object(study_flow, "CONSENT_VERSION", "some-later-version"):
            self.client.get(reverse("web:study_task"))
            session = ParticipantSession.objects.get(pk=self.participant.pk)
            self.assertEqual(
                session.consent_version, stamped,
                "the version they actually agreed to must not be rewritten",
            )
            self.assertEqual(session.stage, ParticipantSession.STAGE_TASK)

    def test_stage_still_never_moves_backwards(self):
        session = study_flow.get_or_create_session(self.participant)
        session.advance_to(ParticipantSession.STAGE_TASK)
        self.assertFalse(session.advance_to(ParticipantSession.STAGE_CONSENT))
        self.assertEqual(session.stage, ParticipantSession.STAGE_TASK)


class TimerTests(StudyFlowTestCase):
    def test_begin_starts_the_timer_and_advances_stage(self):
        self.login_participant()
        resp = self.client.post(reverse("web:study_begin"))
        self.assertRedirects(resp, reverse("web:study_task"), fetch_redirect_response=False)

        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_TASK)
        self.assertIsNotNone(session.task_started_at)
        self.assertIsNotNone(session.task_ends_at)
        self.assertGreater(session.seconds_remaining, 0)

    def test_begin_is_idempotent(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        first = ParticipantSession.objects.get(pk=self.participant.pk).task_ends_at

        self.client.post(reverse("web:study_begin"))
        second = ParticipantSession.objects.get(pk=self.participant.pk).task_ends_at
        self.assertEqual(first, second, "re-POSTing begin must not extend the deadline")

    def test_timer_endpoint_reports_remaining(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        data = self.client.get(reverse("web:study_timer")).json()
        self.assertFalse(data["expired"])
        self.assertGreater(data["remaining"], 0)

    def test_expiry_is_enforced_server_side(self):
        """A participant with JS disabled still cannot keep exploring."""
        self.login_participant()
        self.client.post(reverse("web:study_begin"))

        ParticipantSession.objects.filter(pk=self.participant.pk).update(
            task_ends_at=timezone.now() - dt.timedelta(seconds=1)
        )

        resp = self.client.get(reverse("web:dashboard"), {"model_id": "m_test"})
        self.assertRedirects(resp, reverse("web:study_post"), fetch_redirect_response=False)

        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_POST)
        self.assertFalse(session.finished_early)

    def test_finish_early(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        resp = self.client.post(reverse("web:study_finish"))
        self.assertRedirects(resp, reverse("web:study_post"), fetch_redirect_response=False)

        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_POST)
        self.assertTrue(session.finished_early)
        self.assertIsNotNone(session.task_completed_at)

    def test_stage_never_moves_backwards(self):
        session = ParticipantSession.objects.create(
            user=self.participant, study_group=self.group, stage=ParticipantSession.STAGE_POST
        )
        self.assertFalse(session.advance_to(ParticipantSession.STAGE_TASK))
        self.assertEqual(session.stage, ParticipantSession.STAGE_POST)


class PostStudyTests(StudyFlowTestCase):
    def advance_to_post(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))

    def valid_payload(self, **overrides):
        payload = {
            "tf1": "true",
            "tf2": "false",
            "tf3": "true",
            "tf4": "false",
            "motive": "The settlement ended it.",
            "connections": "Marcos and Priya saw the same approval pattern.",
            "timeline": "Roughly eight months.",
            "threads": "Separate issues.",
            "motive_citations": "doc_1,doc_2",
        }
        payload.update(overrides)
        return payload

    def test_page_renders(self):
        self.advance_to_post()
        resp = self.client.get(reverse("web:study_post"))
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(b"data-answer", resp.content, "answer key must not reach the browser")

    def test_answer_key_is_not_in_the_page(self):
        self.advance_to_post()
        body = self.client.get(reverse("web:study_post")).content.decode()
        for field, correct in study_flow.TF_KEY.items():
            self.assertNotIn(f'data-answer="{correct}"', body)

    def test_submission_scores_and_completes(self):
        self.advance_to_post()
        resp = self.client.post(reverse("web:study_post"), self.valid_payload())
        self.assertRedirects(resp, reverse("web:study_done"), fetch_redirect_response=False)

        response = StudyResponse.objects.get(pk=self.participant.pk)
        self.assertEqual(response.tf_score, 4)
        self.assertEqual(response.study_group, self.group)
        self.assertEqual(response.open_answers["motive"]["citations"], ["doc_1", "doc_2"])
        self.assertIsNotNone(response.task_duration_s)

        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_DONE)

    def test_unsure_is_scored_wrong_but_stored_distinctly(self):
        self.advance_to_post()
        self.client.post(reverse("web:study_post"), self.valid_payload(tf1="unsure"))
        response = StudyResponse.objects.get(pk=self.participant.pk)
        self.assertEqual(response.tf_score, 3)
        self.assertEqual(response.tf_answers["tf1"], "unsure")

    def test_blank_answers_are_accepted_because_the_consent_permits_skipping(self):
        """Replaces a test that asserted the opposite.

        The form used to reject any blank field with a 400. The approved consent
        says "You may skip a question", so refusing to submit contradicted the
        document the participant agreed to. Skipping is now recorded, not
        blocked -- and a skipped field is stored as such rather than dropped.
        """
        self.advance_to_post()
        resp = self.client.post(
            reverse("web:study_post"), self.valid_payload(tf1="", motive="")
        )
        self.assertRedirects(
            resp, reverse("web:study_done"), fetch_redirect_response=False
        )
        response = StudyResponse.objects.get()
        self.assertNotIn("tf1", response.tf_answers)
        self.assertEqual(response.open_answers["motive"]["text"], "")

    def test_an_entirely_blank_submission_is_accepted(self):
        """Someone may decline every question and still complete the study."""
        self.advance_to_post()
        blank = {key: "" for key in self.valid_payload()}
        resp = self.client.post(reverse("web:study_post"), blank)
        self.assertRedirects(
            resp, reverse("web:study_done"), fetch_redirect_response=False
        )
        response = StudyResponse.objects.get()
        self.assertEqual(response.tf_answers, {})
        self.assertEqual(response.tf_score, 0)
        self.assertEqual(response.tf_answered, 0)
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_DONE)

    def test_declining_is_stored_distinctly_from_unsure(self):
        """"Would rather not say" and "don't know" are different findings."""
        self.advance_to_post()
        self.client.post(
            reverse("web:study_post"),
            self.valid_payload(tf1="declined", tf2="unsure"),
        )
        response = StudyResponse.objects.get()
        self.assertEqual(response.tf_answers["tf1"], "declined")
        self.assertEqual(response.tf_answers["tf2"], "unsure")

    def test_answered_count_separates_skipping_from_being_wrong(self):
        self.advance_to_post()
        self.client.post(
            reverse("web:study_post"),
            # tf1 correct, tf2 declined, tf3 skipped, tf4 answered wrongly.
            self.valid_payload(tf1="true", tf2="declined", tf3="", tf4="true"),
        )
        response = StudyResponse.objects.get()
        self.assertEqual(response.tf_score, 1)
        self.assertEqual(response.tf_answered, 2, "declined and blank are not answers")

    def test_unsure_still_counts_as_having_answered(self):
        self.advance_to_post()
        self.client.post(
            reverse("web:study_post"),
            self.valid_payload(tf1="unsure", tf2="unsure", tf3="unsure", tf4="unsure"),
        )
        response = StudyResponse.objects.get()
        self.assertEqual(response.tf_score, 0)
        self.assertEqual(response.tf_answered, 4)

    def test_an_unrecognised_value_is_still_an_error(self):
        """Permissive about absence, not about garbage."""
        self.advance_to_post()
        resp = self.client.post(
            reverse("web:study_post"), self.valid_payload(tf1="maybe")
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(StudyResponse.objects.exists())
        body = resp.content.decode()
        self.assertIn("Marcos and Priya saw the same approval pattern.", body)

    def test_the_form_offers_a_way_to_decline(self):
        self.advance_to_post()
        body = self.client.get(reverse("web:study_post")).content.decode()
        self.assertIn('value="declined"', body)
        self.assertIn("Prefer not to answer", body)

    def test_the_form_warns_against_entering_identifying_information(self):
        self.advance_to_post()
        rendered = visible_text(
            self.client.get(reverse("web:study_post")).content.decode()
        )
        self.assertIn("identifies you", rendered)

    def test_citations_are_optional(self):
        self.advance_to_post()
        payload = self.valid_payload()
        payload.pop("motive_citations")
        resp = self.client.post(reverse("web:study_post"), payload)
        self.assertRedirects(resp, reverse("web:study_done"), fetch_redirect_response=False)

    def test_resubmission_is_blocked(self):
        self.advance_to_post()
        self.client.post(reverse("web:study_post"), self.valid_payload())
        resp = self.client.get(reverse("web:study_post"))
        self.assertRedirects(resp, reverse("web:study_done"), fetch_redirect_response=False)
        self.assertEqual(StudyResponse.objects.count(), 1)

    def test_done_page_renders(self):
        self.advance_to_post()
        self.client.post(reverse("web:study_post"), self.valid_payload())
        self.assertEqual(self.client.get(reverse("web:study_done")).status_code, 200)


class CompletionCodeTests(StudyFlowTestCase):
    """Finishing has to be claimable, or a participant does not get paid."""

    def finish(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        self.client.post(
            reverse("web:study_post"),
            {"tf1": "true", "tf2": "false", "tf3": "true", "tf4": "false",
             "motive": "a", "connections": "b", "timeline": "c", "threads": "d"},
        )

    @override_settings(PROLIFIC_COMPLETION_CODE="C1A2B3C4", PROLIFIC_RETURN_URL="")
    def test_code_is_shown_and_a_return_url_is_derived(self):
        self.finish()
        body = self.client.get(reverse("web:study_done")).content.decode()
        self.assertIn("C1A2B3C4", body)
        self.assertIn("https://app.prolific.com/submissions/complete?cc=C1A2B3C4", body)

    @override_settings(
        PROLIFIC_COMPLETION_CODE="C1A2B3C4",
        PROLIFIC_RETURN_URL="https://app.prolific.com/submissions/complete?cc=OVERRIDE",
    )
    def test_an_explicit_return_url_wins(self):
        self.finish()
        body = self.client.get(reverse("web:study_done")).content.decode()
        self.assertIn("cc=OVERRIDE", body)
        self.assertNotIn("cc=C1A2B3C4", body)

    @override_settings(PROLIFIC_COMPLETION_CODE="", PROLIFIC_RETURN_URL="")
    def test_unconfigured_leaves_the_page_as_it_was(self):
        """The Upwork arm has no completion code; the block must vanish."""
        self.finish()
        body = self.client.get(reverse("web:study_done")).content.decode()
        self.assertNotIn("Confirm your submission", body)
        self.assertIn("close this window", body)
        self.assertNotIn("prolific.com", body)


class TopicRenameTests(StudyFlowTestCase):
    def test_renames_are_private_to_each_participant(self):
        other = User.objects.create_user(
            email="p002@study.local", password="pw", study_group=self.group
        )
        url = reverse("web:rename_topic", args=["m_test", 3])

        with mock.patch.object(
            study_flow, "task_minutes", return_value=30
        ), mock.patch("web.services.runtime._verify_model_ownership", return_value=True):
            self.login_participant()
            self.client.post(reverse("web:study_begin"))
            resp = self.client.post(
                url, data={"new_label": "Approval oversight"}, content_type="application/json"
            )
            self.assertEqual(resp.status_code, 200)

            renames = self.client.get(
                reverse("web:get_topic_renames", args=["m_test"])
            ).json()
            self.assertEqual(renames["topic_labels"], {"3": "Approval oversight"})

            self.client.force_login(other)
            consent_for(other)
            self.client.post(reverse("web:study_begin"))
            other_renames = self.client.get(
                reverse("web:get_topic_renames", args=["m_test"])
            ).json()
            self.assertEqual(
                other_renames["topic_labels"], {}, "renames must not leak between participants"
            )

        self.assertEqual(TopicRename.objects.count(), 1)

    def test_rename_requires_authentication(self):
        resp = self.client.post(
            reverse("web:rename_topic", args=["m_test", 1]),
            data={"new_label": "x"},
            content_type="application/json",
        )
        self.assertIn(resp.status_code, (302, 401))


class InstructionsTests(StudyFlowTestCase):
    def test_instructions_render_with_task_facts(self):
        self.login_participant()
        with mock.patch("web.views._study_corpus_size", return_value=612):
            resp = self.client.get(reverse("web:study_instructions"))
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn("30 min", body)
        self.assertIn("612", body)
        self.assertIn(reverse("web:study_begin"), body)
        # The corrected copy: documents remain available, early finish allowed.
        self.assertNotIn("You won't be able to go back and reopen the documents", body)
        self.assertIn("finish early", body.lower())

    def test_instructions_survive_an_unreachable_backend(self):
        self.login_participant()
        with mock.patch(
            "web.views._study_corpus_size", side_effect=RuntimeError("backend down")
        ):
            resp = self.client.get(reverse("web:study_instructions"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Several hundred", resp.content.decode())


class ParticipantCopyTests(StudyFlowTestCase):
    """Participant-facing copy must not contradict the consent document."""

    def instructions(self):
        self.login_participant()
        with mock.patch("web.views._study_corpus_size", return_value=600):
            return visible_text(
                self.client.get(reverse("web:study_instructions")).content.decode()
            )

    def test_does_not_promise_access_to_emails_during_the_questionnaire(self):
        """The consent says the emails cannot be reopened. This used to say they could."""
        body = self.instructions()
        self.assertNotIn("be able to look documents up", body)
        self.assertIn("won't be able to reopen the emails", body)
        self.assertIn("from memory", body)

    def test_says_that_interactions_are_logged(self):
        self.assertIn("we record how you interact", self.instructions())

    def test_not_sure_is_the_visible_label_for_unsure(self):
        """The approved instrument says "true/false/not-sure"."""
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        body = self.client.get(reverse("web:study_post")).content.decode()
        self.assertIn('value="unsure"', body)
        self.assertIn("Not sure", body)
        self.assertNotIn(">Unsure<", body)


class TimerWidgetTests(StudyFlowTestCase):
    def test_timer_appears_on_the_dashboard_during_the_task(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            body = self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()
        self.assertIn('id="studyTimer"', body)
        self.assertIn(reverse("web:study_finish"), body)

    def test_no_timer_for_non_participants(self):
        self.client.force_login(self.outsider)
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            body = self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()
        self.assertNotIn('id="studyTimer"', body)


class StudyDocsTests(StudyFlowTestCase):
    BASE_DOCS = [
        {"id": "d1", "display_id": "1592", "subject": "Board Settlement", "date": "2025-09-24"},
        {"id": "d2", "display_id": "1452", "subject": "Decoded Slack", "date": "2025-09-25"},
        {"id": "d3", "display_id": "1404", "subject": "Routine Assay Log", "date": "2025-09-25"},
    ]
    TOPIC_LABEL = {"d1": "Settlement", "d2": "Investigation", "d3": "Investigation"}
    TOPIC_ID = {"d1": "0", "d2": "1", "d3": "1"}

    def advance_to_post(self, user=None):
        subject = user or self.participant
        self.client.force_login(subject)
        consent_for(subject)
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))

    def patch_base(self):
        return mock.patch(
            "web.views._study_base_documents",
            return_value=(self.BASE_DOCS, self.TOPIC_LABEL, self.TOPIC_ID, None),
        )

    def test_falls_back_to_model_topic_labels(self):
        self.advance_to_post()
        with self.patch_base(), mock.patch("web.views.requests.get") as get:
            get.return_value = mock.Mock(
                content=b"{}", json=lambda: {"labels": {}}, raise_for_status=lambda: None
            )
            data = self.client.get(reverse("web:study_docs")).json()

        self.assertEqual(sorted(data["themes"]), ["Investigation", "Settlement"])
        themes = {e["id"]: e["theme"] for e in data["emails"]}
        self.assertEqual(themes, self.TOPIC_LABEL)

    def test_rename_overrides_the_model_label(self):
        self.advance_to_post()
        TopicRename.objects.create(
            user=self.participant, model_id="m_test", topic_id="1", label="My investigation"
        )
        with self.patch_base(), mock.patch("web.views.requests.get") as get:
            get.return_value = mock.Mock(
                content=b"{}", json=lambda: {"labels": {}}, raise_for_status=lambda: None
            )
            data = self.client.get(reverse("web:study_docs")).json()

        themes = {e["id"]: e["theme"] for e in data["emails"]}
        self.assertEqual(themes["d2"], "My investigation")
        self.assertEqual(themes["d3"], "My investigation")
        self.assertEqual(themes["d1"], "Settlement")

    def test_active_learning_label_beats_a_rename(self):
        self.advance_to_post()
        TopicRename.objects.create(
            user=self.participant, model_id="m_test", topic_id="1", label="My investigation"
        )
        with self.patch_base(), mock.patch("web.views.requests.get") as get:
            get.return_value = mock.Mock(
                content=b'{"labels": {}}',
                json=lambda: {"labels": {"d2": "Data tampering"}},
                raise_for_status=lambda: None,
            )
            data = self.client.get(reverse("web:study_docs")).json()

        themes = {e["id"]: e["theme"] for e in data["emails"]}
        self.assertEqual(themes["d2"], "Data tampering", "AL label must win")
        self.assertEqual(themes["d3"], "My investigation", "rename still applies elsewhere")

    def test_themes_do_not_leak_between_participants(self):
        other = User.objects.create_user(
            email="p003@study.local", password="pw", study_group=self.group
        )
        TopicRename.objects.create(
            user=self.participant, model_id="m_test", topic_id="1", label="Mine only"
        )
        self.advance_to_post(other)
        with self.patch_base(), mock.patch("web.views.requests.get") as get:
            get.return_value = mock.Mock(
                content=b"{}", json=lambda: {"labels": {}}, raise_for_status=lambda: None
            )
            data = self.client.get(reverse("web:study_docs")).json()

        self.assertNotIn("Mine only", data["themes"])

    def test_unresolved_documents_land_in_unclassified_last(self):
        self.advance_to_post()
        with mock.patch(
            "web.views._study_base_documents",
            return_value=(self.BASE_DOCS, {"d1": "Settlement"}, {"d1": "0"}, None),
        ), mock.patch("web.views.requests.get") as get:
            get.return_value = mock.Mock(
                content=b"{}", json=lambda: {"labels": {}}, raise_for_status=lambda: None
            )
            data = self.client.get(reverse("web:study_docs")).json()

        self.assertEqual(data["themes"][-1], study_flow.UNCLASSIFIED_THEME)
        themes = {e["id"]: e["theme"] for e in data["emails"]}
        self.assertEqual(themes["d2"], study_flow.UNCLASSIFIED_THEME)
        self.assertEqual(len(data["emails"]), 3, "no document may be dropped")

    def test_no_active_learning_call_when_the_feature_is_off(self):
        self.group.enable_active_learning = False
        self.group.save()
        self.advance_to_post()
        with self.patch_base(), mock.patch("web.views.requests.get") as get:
            self.client.get(reverse("web:study_docs"))
        get.assert_not_called()


class EmailMetaTests(TestCase):
    def test_parses_rfc822_headers_from_raw_text(self):
        from web.views import _study_email_meta

        subject, date = _study_email_meta(
            {"raw_text": "From: a@b.c\nSubject: Board Settlement\nDate: 2025-09-24\n\nBody"}
        )
        self.assertEqual(subject, "Board Settlement")
        self.assertEqual(date, "2025-09-24")

    def test_explicit_fields_win(self):
        from web.views import _study_email_meta

        subject, date = _study_email_meta(
            {"subject": "Real", "date": "2026-01-01", "raw_text": "Subject: Other"}
        )
        self.assertEqual((subject, date), ("Real", "2026-01-01"))

    def test_falls_back_to_first_line_when_there_are_no_headers(self):
        from web.views import _study_email_meta

        subject, date = _study_email_meta({"text": "wilderness act sec designates areas"})
        self.assertEqual(subject, "wilderness act sec designates areas")
        self.assertEqual(date, "")


class TimerHandoffTests(StudyFlowTestCase):
    def test_timer_endpoint_hands_off_cleanly_after_expiry(self):
        """The countdown widget must get JSON, not an HTML redirect, at the boundary."""
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        ParticipantSession.objects.filter(pk=self.participant.pk).update(
            task_ends_at=timezone.now() - dt.timedelta(seconds=1)
        )

        resp = self.client.get(reverse("web:study_timer"))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["expired"])
        self.assertEqual(data["remaining"], 0)
        self.assertEqual(data["redirect"], reverse("web:study_post"))

    def test_finish_is_idempotent_after_the_stage_has_moved_on(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        completed = ParticipantSession.objects.get(pk=self.participant.pk).task_completed_at

        resp = self.client.post(
            reverse("web:study_finish"), HTTP_X_REQUESTED_WITH="XMLHttpRequest"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["redirect"], reverse("web:study_post"))
        self.assertEqual(
            ParticipantSession.objects.get(pk=self.participant.pk).task_completed_at,
            completed,
            "a second finish must not rewrite the completion time",
        )


class PerUserDashboardTests(TestCase):
    """Dashboards are assigned per user; the grouping is who shares one."""

    def setUp(self):
        from web.study_features import (
            FEATURE_ACTIVE_LEARNING,
            get_study_model_id,
            is_study_participant,
            user_has_feature,
        )
        self.get_model = get_study_model_id
        self.is_participant = is_study_participant
        self.has_al = lambda u: user_has_feature(u, FEATURE_ACTIVE_LEARNING)

    def make(self, **kwargs):
        email = kwargs.pop("email", f"u{User.objects.count()}@study.local")
        return User.objects.create_user(email=email, password="pw", **kwargs)

    def test_assigning_a_dashboard_makes_someone_a_participant(self):
        u = self.make()
        self.assertFalse(self.is_participant(u))

        u.study_model_id = "m_alpha"
        u.save()
        self.assertTrue(self.is_participant(u))
        self.assertEqual(self.get_model(u), "m_alpha")

    def test_per_user_dashboard_overrides_the_group(self):
        g = StudyGroup.objects.create(name="G", slug="g", default_model_id="m_group")
        u = self.make(study_group=g, study_model_id="m_own")
        self.assertEqual(self.get_model(u), "m_own")

    def test_group_dashboard_still_works_as_a_fallback(self):
        g = StudyGroup.objects.create(name="G2", slug="g2", default_model_id="m_group")
        u = self.make(study_group=g)
        self.assertEqual(self.get_model(u), "m_group")
        self.assertTrue(self.is_participant(u))

    def test_active_learning_is_on_by_default(self):
        u = self.make(study_model_id="m_alpha")
        self.assertIsNone(u.enable_active_learning)
        self.assertTrue(self.has_al(u), "active learning must default to on")

    def test_admin_can_switch_active_learning_off_per_user(self):
        u = self.make(study_model_id="m_alpha", enable_active_learning=False)
        self.assertFalse(self.has_al(u))

    def test_per_user_switch_overrides_the_group(self):
        g = StudyGroup.objects.create(
            name="G3", slug="g3", default_model_id="m_group", enable_active_learning=False
        )
        on = self.make(email="on@s.local", study_group=g, enable_active_learning=True)
        inherit = self.make(email="inherit@s.local", study_group=g)
        self.assertTrue(self.has_al(on), "explicit per-user True must win")
        self.assertFalse(self.has_al(inherit), "unset must inherit the group's False")

    def test_non_participants_keep_full_access(self):
        u = self.make()
        self.assertTrue(self.has_al(u))

    def test_grouping_is_derived_from_the_dashboard(self):
        a1 = self.make(email="a1@s.local", study_model_id="m_alpha")
        a2 = self.make(email="a2@s.local", study_model_id="m_alpha")
        b1 = self.make(email="b1@s.local", study_model_id="m_beta")
        self.assertEqual(
            set(User.objects.filter(study_model_id="m_alpha").values_list("email", flat=True)),
            {a1.email, a2.email},
        )
        self.assertEqual(User.objects.filter(study_model_id="m_beta").count(), 1)
        self.assertEqual(b1.study_model_id, "m_beta")


class ParticipantCodeTests(TestCase):
    """The consent says data is "linked only to a study-specific code"."""

    def test_generated_logins_get_a_readable_code(self):
        from web.admin import participant_code_for

        self.assertEqual(participant_code_for("p001@study.local"), "p001")
        self.assertEqual(participant_code_for("P042@STUDY.LOCAL"), "p042")

    def test_real_addresses_get_no_code(self):
        """A work address's local part is usually a name -- not a de-identifier."""
        from web.admin import participant_code_for

        self.assertIsNone(participant_code_for("ana.lopez@lab.gov"))
        self.assertIsNone(participant_code_for("not-an-email"))
        self.assertIsNone(participant_code_for(""))

    def test_export_id_prefers_the_code(self):
        u = User.objects.create_user(
            email="p007@study.local", password="pw", participant_code="p007"
        )
        self.assertEqual(u.export_id, "p007")

    def test_export_id_never_falls_back_to_email(self):
        u = User.objects.create_user(email="ana.lopez@lab.gov", password="pw")
        self.assertEqual(u.export_id, str(u.pk))
        self.assertNotIn("@", u.export_id)
        self.assertNotIn("ana", u.export_id)

    def test_codes_are_unique(self):
        from django.db import IntegrityError

        User.objects.create_user(
            email="p001@study.local", password="pw", participant_code="p001"
        )
        with self.assertRaises(IntegrityError):
            User.objects.create_user(
                email="other@study.local", password="pw", participant_code="p001"
            )

    def test_blank_codes_do_not_collide(self):
        """Unique + null: many staff accounts have no code at all."""
        User.objects.create_user(email="a@x.com", password="pw")
        User.objects.create_user(email="b@x.com", password="pw")
        self.assertEqual(User.objects.filter(participant_code__isnull=True).count(), 2)


class ConditionSnapshotTests(StudyFlowTestCase):
    def test_flipping_active_learning_mid_run_does_not_change_the_condition(self):
        self.participant.study_model_id = "m_test"
        self.participant.enable_active_learning = True
        self.participant.save()

        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.assertTrue(ParticipantSession.objects.get(pk=self.participant.pk).active_learning)
        self.assertEqual(
            self.client.get(
                reverse("web:active_learning_two"), {"model_id": "m_test"}
            ).status_code,
            200,
        )

        # Admin flips the switch while the participant is mid-task.
        self.participant.enable_active_learning = False
        self.participant.save()

        self.assertEqual(
            self.client.get(
                reverse("web:active_learning_two"), {"model_id": "m_test"}
            ).status_code,
            200,
            "a run already underway must keep the condition it started in",
        )

    def test_dashboard_is_snapshotted_at_task_start(self):
        self.participant.study_model_id = "m_first"
        self.participant.save()
        self.login_participant()
        self.client.post(reverse("web:study_begin"))

        self.participant.study_model_id = "m_second"
        self.participant.save()

        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.model_id, "m_first")


class AssignDashboardActionTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.u1 = User.objects.create_user(email="a@s.local", password="pw")
        self.u2 = User.objects.create_user(email="b@s.local", password="pw")
        self.url = reverse("admin:web_user_changelist")

    def post_action(self, extra=None):
        data = {
            "action": "assign_dashboard",
            "_selected_action": [str(self.u1.pk), str(self.u2.pk)],
        }
        data.update(extra or {})
        return self.client.post(self.url, data, follow=False)

    def test_action_shows_a_confirmation_page(self):
        resp = self.post_action()
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Assign dashboard")
        self.assertContains(resp, "a@s.local")

    def test_applying_assigns_the_dashboard_to_everyone_selected(self):
        with mock.patch(
            "web.admin.available_dashboards", return_value=[("m_alpha", "Alpha")]
        ):
            resp = self.post_action({"apply": "1", "study_model_id": "m_alpha",
                                     "active_learning": "keep"})
        self.assertEqual(resp.status_code, 302)
        self.u1.refresh_from_db()
        self.u2.refresh_from_db()
        self.assertEqual(self.u1.study_model_id, "m_alpha")
        self.assertEqual(self.u2.study_model_id, "m_alpha")
        self.assertIsNone(self.u1.enable_active_learning, "'keep' must not touch the switch")

    def test_action_can_switch_active_learning_off(self):
        with mock.patch(
            "web.admin.available_dashboards", return_value=[("m_alpha", "Alpha")]
        ):
            self.post_action({"apply": "1", "study_model_id": "m_alpha",
                              "active_learning": "off"})
        self.u1.refresh_from_db()
        self.assertFalse(self.u1.enable_active_learning)

    def test_reassignment_clears_a_run_that_has_not_started(self):
        ParticipantSession.objects.create(user=self.u1, model_id="m_old")
        started = ParticipantSession.objects.create(
            user=self.u2, model_id="m_old", task_started_at=timezone.now()
        )
        with mock.patch(
            "web.admin.available_dashboards", return_value=[("m_alpha", "Alpha")]
        ):
            self.post_action({"apply": "1", "study_model_id": "m_alpha",
                              "active_learning": "keep"})

        self.assertFalse(ParticipantSession.objects.filter(pk=self.u1.pk).exists())
        self.assertTrue(
            ParticipantSession.objects.filter(pk=self.u2.pk).exists(),
            "a run already in progress must not be wiped by a reassignment",
        )
        started.refresh_from_db()
        self.assertEqual(started.model_id, "m_old")

    def test_only_one_action_regardless_of_group_count(self):
        for i in range(5):
            StudyGroup.objects.create(name=f"G{i}", slug=f"g{i}")
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(self.url)
        actions = resp.context["action_form"].fields["action"].choices
        assign = [a for a, _label in actions if a and "assign" in a]
        self.assertEqual(assign, ["assign_dashboard"])


class AdminCreateParticipantsTests(TestCase):
    """The bulk-create page in Django admin."""

    DASHBOARDS = [("m_alpha", "model1  (m_alpha)"), ("m_beta", "model2  (m_beta)")]

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.url = reverse("admin:web_user_create_participants")

    def patched(self):
        return mock.patch("web.admin.available_dashboards", return_value=self.DASHBOARDS)

    def payload(self, **overrides):
        data = {
            "count": "6",
            "emails": "",
            "prefix": "p",
            "domain": "study.local",
            "start": "1",
            "dashboards": ["m_alpha", "m_beta"],
            "active_learning": "on",
        }
        data.update(overrides)
        return data

    def test_page_is_reachable_and_lists_dashboards(self):
        with self.patched():
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "m_alpha")
        self.assertContains(resp, "Create participants")

    def test_link_appears_on_the_user_changelist(self):
        with self.patched():
            resp = self.client.get(reverse("admin:web_user_changelist"))
        self.assertContains(resp, self.url)

    def test_generates_accounts_spread_across_dashboards(self):
        with self.patched():
            resp = self.client.post(self.url, self.payload())
        self.assertEqual(resp.status_code, 200)

        created = User.objects.filter(email__endswith="@study.local").order_by("email")
        self.assertEqual(created.count(), 6)
        self.assertEqual(
            list(created.values_list("email", flat=True)),
            [f"p{n:03d}@study.local" for n in range(1, 7)],
        )
        self.assertEqual(created.filter(study_model_id="m_alpha").count(), 3)
        self.assertEqual(created.filter(study_model_id="m_beta").count(), 3)

    def test_passwords_are_shown_once_and_actually_work(self):
        with self.patched():
            resp = self.client.post(self.url, self.payload(count="1"))
        row = resp.context["rows"][0]
        self.assertTrue(row["password"], "a password must be shown")
        self.assertContains(resp, row["password"])

        user = User.objects.get(email=row["email"])
        self.assertTrue(
            user.check_password(row["password"]), "the shown password must be the real one"
        )
        self.assertNotEqual(user.password, row["password"], "must be stored hashed")

    def test_pasted_emails_are_used_verbatim(self):
        with self.patched():
            self.client.post(
                self.url,
                self.payload(count="", emails="Ana@Lab.gov\nben@lab.gov, cara@lab.gov"),
            )
        self.assertEqual(
            set(User.objects.exclude(email="root@example.com").values_list("email", flat=True)),
            {"ana@lab.gov", "ben@lab.gov", "cara@lab.gov"},
        )

    def test_active_learning_can_be_switched_off(self):
        with self.patched():
            self.client.post(self.url, self.payload(count="2", active_learning="off"))
        for u in User.objects.filter(email__endswith="@study.local"):
            self.assertFalse(u.enable_active_learning)

    def test_count_and_emails_are_mutually_exclusive(self):
        with self.patched():
            resp = self.client.post(self.url, self.payload(emails="a@b.com"))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(User.objects.filter(email="a@b.com").exists())
        self.assertContains(resp, "not both")

    def test_rejects_input_that_is_not_an_email(self):
        with self.patched():
            resp = self.client.post(self.url, self.payload(count="", emails="not-an-email"))
        self.assertContains(resp, "do not look like email addresses")
        self.assertEqual(User.objects.count(), 1)

    def test_existing_accounts_are_skipped_by_default(self):
        existing = User.objects.create_user(email="p001@study.local", password="original")
        with self.patched():
            resp = self.client.post(self.url, self.payload(count="2", skip_existing="on"))

        existing.refresh_from_db()
        self.assertTrue(existing.check_password("original"), "must not reset their password")
        self.assertEqual(existing.study_model_id, "", "must not reassign them")
        self.assertEqual(resp.context["skipped"], 1)
        self.assertEqual(resp.context["created"], 1)

    def test_existing_accounts_are_reassigned_when_not_skipping(self):
        existing = User.objects.create_user(email="p001@study.local", password="original")
        with self.patched():
            resp = self.client.post(self.url, self.payload(count="1"))
        existing.refresh_from_db()
        self.assertEqual(existing.study_model_id, "m_alpha")
        self.assertFalse(existing.check_password("original"))
        self.assertEqual(resp.context["updated"], 1)

    def test_start_offset_avoids_collisions_with_an_earlier_cohort(self):
        with self.patched():
            self.client.post(self.url, self.payload(count="3"))
            self.client.post(self.url, self.payload(count="3", start="4"))
        emails = set(User.objects.filter(email__endswith="@study.local").values_list("email", flat=True))
        self.assertEqual(emails, {f"p{n:03d}@study.local" for n in range(1, 7)})

    def test_a_run_already_underway_is_not_disturbed(self):
        user = User.objects.create_user(email="p001@study.local", password="pw")
        session = ParticipantSession.objects.create(
            user=user, model_id="m_old", task_started_at=timezone.now()
        )
        with self.patched():
            self.client.post(self.url, self.payload(count="1", skip_existing=""))
        session.refresh_from_db()
        self.assertEqual(session.model_id, "m_old")

    def test_non_staff_cannot_reach_the_page(self):
        self.client.force_login(User.objects.create_user(email="joe@x.com", password="pw"))
        resp = self.client.get(self.url)
        self.assertIn(resp.status_code, (302, 403))

    def test_add_user_form_offers_the_dashboard_picker(self):
        with self.patched():
            resp = self.client.get(reverse("admin:web_user_add"))
        self.assertContains(resp, "study_model_id")
        self.assertContains(resp, "m_alpha")


class GroupMembershipAdminTests(TestCase):
    """Browsing and editing group membership from the group page."""

    DASHBOARDS = [("m_alpha", "model1  (m_alpha)"), ("m_beta", "model2  (m_beta)")]

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.group = StudyGroup.objects.create(
            name="Condition A", slug="cond-a", default_model_id="m_alpha"
        )
        self.other = StudyGroup.objects.create(name="Condition B", slug="cond-b")
        self.u1 = User.objects.create_user(email="a@s.local", password="pw")
        self.u2 = User.objects.create_user(email="b@s.local", password="pw")
        self.change_url = reverse("admin:web_studygroup_change", args=[self.group.pk])

    def patched(self):
        return mock.patch("web.admin.available_dashboards", return_value=self.DASHBOARDS)

    def form_data(self, members):
        return {
            "name": self.group.name,
            "slug": self.group.slug,
            "description": "",
            "default_model_id": self.group.default_model_id,
            "enable_active_learning": "on",
            "members": [str(pk) for pk in members],
        }

    def test_group_page_shows_a_members_picker(self):
        with self.patched():
            resp = self.client.get(self.change_url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "members")
        self.assertContains(resp, "a@s.local")

    def test_user_count_links_to_the_filtered_user_list(self):
        self.u1.study_group = self.group
        self.u1.save()
        with self.patched():
            resp = self.client.get(reverse("admin:web_studygroup_changelist"))
        self.assertContains(resp, f"study_group__id__exact={self.group.pk}")
        self.assertContains(resp, "1 user")

    def test_adding_members_from_the_group_page(self):
        with self.patched():
            resp = self.client.post(self.change_url, self.form_data([self.u1.pk, self.u2.pk]))
        self.assertEqual(resp.status_code, 302)
        self.u1.refresh_from_db()
        self.u2.refresh_from_db()
        self.assertEqual(self.u1.study_group, self.group)
        self.assertEqual(self.u2.study_group, self.group)

    def test_removing_a_member_clears_only_their_group(self):
        self.u1.study_group = self.group
        self.u1.study_model_id = "m_beta"
        self.u1.save()

        with self.patched():
            self.client.post(self.change_url, self.form_data([]))

        self.u1.refresh_from_db()
        self.assertIsNone(self.u1.study_group)
        self.assertEqual(
            self.u1.study_model_id, "m_beta", "a directly-assigned dashboard must survive"
        )

    def test_members_of_another_group_are_not_disturbed(self):
        self.u2.study_group = self.other
        self.u2.save()
        with self.patched():
            self.client.post(self.change_url, self.form_data([self.u1.pk]))
        self.u2.refresh_from_db()
        self.assertEqual(self.u2.study_group, self.other)

    def test_member_table_shows_where_each_dashboard_comes_from(self):
        self.u1.study_group = self.group           # inherits m_alpha
        self.u1.save()
        self.u2.study_group = self.group
        self.u2.study_model_id = "m_beta"          # own override
        self.u2.save()

        with self.patched():
            body = self.client.get(self.change_url).content.decode()

        self.assertIn("from group", body)
        self.assertIn("model1", body)
        self.assertIn("model2", body)

    def test_staff_are_not_offered_as_members(self):
        with self.patched():
            resp = self.client.get(self.change_url)
        form = resp.context["adminform"].form
        offered = set(form.fields["members"].queryset.values_list("email", flat=True))
        self.assertNotIn("root@example.com", offered)
        self.assertEqual(offered, {"a@s.local", "b@s.local"})

    def test_user_changelist_filters_by_dashboard(self):
        self.u1.study_model_id = "m_alpha"
        self.u1.save()
        self.u2.study_model_id = "m_beta"
        self.u2.save()
        with self.patched():
            resp = self.client.get(
                reverse("admin:web_user_changelist"), {"study_model_id": "m_alpha"}
            )
        shown = [u.email for u in resp.context["cl"].result_list]
        self.assertEqual(shown, ["a@s.local"])


class StudyLoginTests(StudyFlowTestCase):
    def test_study_login_page_renders_without_signup_or_sso(self):
        body = self.client.get(reverse("web:study_login")).content.decode()
        self.assertIn("Sign in to begin", body)
        self.assertNotIn(reverse("web:signup"), body)
        self.assertNotIn("Okta", body)

    def test_same_page_regardless_of_condition(self):
        """A separate door per arm would signal the condition before they start."""
        import re

        def body(resp):
            # The CSRF token rotates per request; everything else must match.
            return re.sub(rb'value="[^"]{32,}"', b'value="TOKEN"', resp.content)

        a = body(self.client.get(reverse("web:study_login")))
        self.group.enable_active_learning = False
        self.group.save()
        b = body(self.client.get(reverse("web:study_login")))
        self.assertEqual(a, b)

    def test_signing_in_lands_on_the_current_stage(self):
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "pw"},
        )
        self.assertRedirects(
            resp, reverse("web:study_consent"), fetch_redirect_response=False
        )

    def test_signing_in_again_lands_on_the_stage_they_reached(self):
        consent_for(self.participant)
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "pw"},
        )
        self.assertRedirects(
            resp, reverse("web:study_instructions"), fetch_redirect_response=False
        )

    def test_bad_password_returns_to_the_study_login_not_the_main_one(self):
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "wrong"},
        )
        self.assertRedirects(
            resp, reverse("web:study_login"), fetch_redirect_response=False
        )

    def test_reachable_while_signed_out_at_any_stage(self):
        self.assertEqual(self.client.get(reverse("web:study_login")).status_code, 200)

    def test_already_signed_in_goes_straight_to_their_stage(self):
        self.login_participant()
        resp = self.client.get(reverse("web:study_login"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/study/", resp["Location"])

    def test_staff_can_use_it_too_and_are_not_gated(self):
        resp = self.client.post(
            reverse("web:study_login"), {"email": self.staff.email, "password": "pw"}
        )
        self.assertEqual(resp.status_code, 302)
        self.assertNotIn("/study/", resp["Location"])


class SignupGateTests(TestCase):
    def test_signup_is_open_by_default(self):
        self.assertEqual(self.client.get(reverse("web:signup")).status_code, 200)

    def test_signup_can_be_closed_for_a_study_deployment(self):
        with self.settings(TOVA_ALLOW_SIGNUP=False):
            resp = self.client.get(reverse("web:signup"))
            self.assertRedirects(
                resp, reverse("web:login"), fetch_redirect_response=False
            )
            self.client.post(
                reverse("web:signup"),
                {"email": "sneaky@x.com", "password": "pw12345678", "name": "S"},
            )
        self.assertFalse(
            User.objects.filter(email="sneaky@x.com").exists(),
            "a closed signup must not create accounts that bypass study gating",
        )


class NavbarTimerTests(StudyFlowTestCase):
    """The timer lives in the navbar; study pages drop irrelevant nav items."""

    def dashboard_body(self):
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            return self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()

    def test_timer_sits_inside_the_navbar(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        body = self.dashboard_body()

        self.assertIn('id="studyTimer"', body)
        # Between the nav links and the account menu, so the meter can grow into
        # the free space rather than the nav list's auto margin swallowing it.
        self.assertLess(
            body.index("</ul>"), body.index('id="studyTimer"'),
            "the timer must sit after the nav list, not floating in <body>",
        )
        self.assertLess(
            body.index('id="studyTimer"'),
            body.index('<div class="d-flex align-items-center">'),
            "the timer must come before the account menu cluster",
        )
        self.assertIn("me-3", body)
        self.assertNotIn('navbar-nav me-auto', body)

    def test_timer_carries_the_totals_its_meter_needs(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        body = self.dashboard_body()
        self.assertIn('data-total="1800"', body)
        self.assertIn("studyTimerFill", body)
        self.assertRegex(body, r'data-remaining="\d+"')

    def test_trained_models_is_hidden_from_participants(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        body = self.dashboard_body()
        # Assert on the link, not the phrase: base.html also mentions it in an
        # orphaned help modal that has no trigger.
        self.assertNotIn(f'href="{reverse("web:trained_models")}"', body)
        self.assertNotIn(f'href="{reverse("web:load_data_page")}"', body)

    def test_nav_list_keeps_its_auto_margin_for_everyone_else(self):
        """Only a running task changes the navbar's spacing."""
        self.client.force_login(self.staff)
        body = self.dashboard_body()
        self.assertIn("navbar-nav me-auto", body)

    def test_trained_models_still_there_for_everyone_else(self):
        self.client.force_login(self.staff)
        body = self.dashboard_body()
        self.assertIn(f'href="{reverse("web:trained_models")}"', body)
        self.assertNotIn('id="studyTimer"', body)

    def test_timer_appears_on_base_template_pages_too(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        body = self.client.get(
            reverse("web:active_learning_two"), {"model_id": "m_test"}
        ).content.decode()
        self.assertIn('id="studyTimer"', body)
        self.assertNotIn(f'href="{reverse("web:trained_models")}"', body)

    def test_timer_is_rendered_once_not_twice(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.assertEqual(self.dashboard_body().count('id="studyTimer"'), 1)

    def test_participants_cannot_reach_the_model_list_by_url(self):
        """Hiding the button is only half of it."""
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        resp = self.client.get(reverse("web:trained_models"))
        self.assertEqual(resp.status_code, 302)
        self.assertNotIn("trained-models", resp["Location"])

    def test_no_timer_before_the_task_starts(self):
        self.login_participant()
        body = self.client.get(reverse("web:study_instructions")).content.decode()
        self.assertNotIn('id="studyTimer"', body)


class ParticipantAccountMenuTests(StudyFlowTestCase):
    """Participants get Finish where the account menu used to be."""

    def start_task(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))

    def al_body(self):
        return self.client.get(
            reverse("web:active_learning_two"), {"model_id": "m_test"}
        ).content.decode()

    def dashboard_body(self):
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            return self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()

    # The trigger button, not the CSS class -- ".tova-profile-trigger" also
    # appears in the stylesheet, which would make these pass either way.
    TRIGGER = 'aria-label="Account menu"'
    LLM_ITEM = 'class="dropdown-item" data-bs-toggle="modal" data-bs-target="#llmSettingsModal"'
    CFG_ITEM = 'class="dropdown-item" data-bs-toggle="modal" data-bs-target="#configModal"'

    def test_account_menu_is_gone_from_the_active_learning_page(self):
        self.start_task()
        body = self.al_body()
        self.assertNotIn(self.TRIGGER, body)
        self.assertNotIn("Settings &amp; sign out", body)

    def test_finish_is_there_instead(self):
        self.start_task()
        self.assertIn('id="studyTimerFinish"', self.al_body())

    def test_settings_cannot_be_opened_mid_study(self):
        """LLM settings and the config editor would change behaviour mid-run."""
        self.start_task()
        body = self.al_body()
        self.assertNotIn(self.LLM_ITEM, body)
        self.assertNotIn(self.CFG_ITEM, body)

    def test_saving_llm_config_is_refused_mid_study(self):
        """Belt and braces: the modal markup is still in the DOM."""
        self.start_task()
        resp = self.client.post(
            reverse("web:save_llm_config"), data="{}", content_type="application/json"
        )
        self.assertIn(resp.status_code, (302, 409))

    def test_account_menu_also_gone_from_the_main_dashboard(self):
        self.start_task()
        self.assertNotIn(self.TRIGGER, self.dashboard_body())

    def test_everyone_else_keeps_their_account_menu(self):
        self.client.force_login(self.staff)
        self.assertIn(self.TRIGGER, self.dashboard_body())
        self.assertIn(
            self.TRIGGER,
            self.client.get(reverse("web:trained_models")).content.decode(),
        )

    def test_logout_still_works_for_a_participant(self):
        """No button, but the session runner can still reset the machine."""
        self.start_task()
        self.assertEqual(self.client.get(reverse("web:logout")).status_code, 302)
        self.assertNotIn("_auth_user_id", self.client.session)


class ParticipantLogoutTests(StudyFlowTestCase):
    def test_sign_out_button_is_on_the_task_pages(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            body = self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()
        self.assertIn("study-timer__logout", body)
        self.assertIn(f'href="{reverse("web:logout")}"', body)

    def test_participant_logout_lands_on_the_study_login(self):
        self.login_participant()
        resp = self.client.get(reverse("web:logout"))
        self.assertRedirects(
            resp, reverse("web:study_login"), fetch_redirect_response=False
        )
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_staff_logout_still_uses_the_main_login(self):
        self.client.force_login(self.staff)
        resp = self.client.get(reverse("web:logout"))
        self.assertRedirects(resp, reverse("web:login"), fetch_redirect_response=False)


class InteractionLoggingTests(StudyFlowTestCase):
    URL = None

    def setUp(self):
        super().setUp()
        self.URL = reverse("web:study_events")
        self.participant.study_model_id = "m_test"
        self.participant.save()

    def post_events(self, events):
        return self.client.post(
            self.URL, data=json.dumps({"events": events}), content_type="application/json"
        )

    def test_collector_is_present_for_participants(self):
        self.login_participant()
        body = self.client.get(reverse("web:study_instructions")).content.decode()
        self.assertIn("__tovaTelemetry", body)
        self.assertIn(self.URL, body)

    def test_collector_is_absent_for_everyone_else(self):
        self.client.force_login(self.staff)
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            body = self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()
        self.assertNotIn("__tovaTelemetry", body)

    def test_events_are_stored_with_their_modality(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        resp = self.post_events([
            {"action": "click", "modality": "mouse", "target": "#docTable", "t": 1200},
            {"action": "shortcut", "modality": "keyboard", "target": "#al2Doc",
             "detail": {"key": "Enter"}, "t": 1400},
        ])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["stored"], 2)

        events = list(InteractionEvent.objects.filter(user=self.participant))
        self.assertEqual({e.modality for e in events}, {"mouse", "keyboard"})
        self.assertEqual(events[0].model_id, "m_test")
        self.assertEqual(events[0].stage, ParticipantSession.STAGE_TASK)

    def test_unknown_actions_and_modalities_are_not_trusted(self):
        self.login_participant()
        self.post_events([
            {"action": "exfiltrate", "modality": "mouse"},
            {"action": "click", "modality": "telepathy", "target": "x"},
        ])
        events = list(InteractionEvent.objects.all())
        self.assertEqual(len(events), 1, "unknown actions must be dropped")
        self.assertEqual(events[0].modality, "other", "unknown modality falls back")

    def test_batch_size_is_capped(self):
        self.login_participant()
        self.post_events([{"action": "click", "modality": "mouse"}] * 500)
        self.assertEqual(InteractionEvent.objects.count(), 200)

    def test_malformed_payloads_are_rejected(self):
        self.login_participant()
        resp = self.client.post(
            self.URL, data=json.dumps({"events": "nope"}), content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)

    def test_staff_events_are_discarded_not_stored(self):
        self.client.force_login(self.staff)
        resp = self.post_events([{"action": "click", "modality": "mouse"}])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InteractionEvent.objects.count(), 0)

    def test_typed_content_is_never_stored(self):
        """The collector sends 'char'; nothing should carry what was typed."""
        self.login_participant()
        self.post_events([
            {"action": "key", "modality": "keyboard", "target": "textarea[text]",
             "detail": {"key": "char"}},
        ])
        event = InteractionEvent.objects.get()
        self.assertEqual(event.detail, {"key": "char"})

    def test_collector_substitutes_char_for_printable_keys(self):
        """Guards the one line that keeps this from being a keylogger."""
        body = read_template_source("study/_telemetry.html")
        branch = body[body.index("} else if (key.length === 1) {"):]
        branch = branch[: branch.index("} else {")]
        self.assertIn("label = 'char';", branch)
        self.assertNotIn("label = key", branch)


class KeyboardMouseComparisonTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.p1 = User.objects.create_user(
            email="p1@study.local", password="pw", study_model_id="m_alpha",
            participant_code="p1",
        )
        self.p2 = User.objects.create_user(
            email="p2@study.local", password="pw", study_model_id="m_beta",
            participant_code="p2",
        )
        mk = InteractionEvent.objects.create
        for _ in range(3):
            mk(user=self.p1, model_id="m_alpha", action="click", modality="mouse")
        for _ in range(1):
            mk(user=self.p1, model_id="m_alpha", action="shortcut", modality="keyboard",
               detail={"key": "Enter"})
        for _ in range(1):
            mk(user=self.p2, model_id="m_beta", action="click", modality="mouse")
        for _ in range(3):
            mk(user=self.p2, model_id="m_beta", action="shortcut", modality="keyboard",
               detail={"key": "ArrowRight"})
        self.url = reverse("admin:web_interactionevent_comparison")

    def test_comparison_reports_shares_per_dashboard(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        shares = {r["key"]: r["keyboard_share"] for r in resp.context["by_dashboard"]}
        self.assertEqual(shares["m_alpha"], 25.0)
        self.assertEqual(shares["m_beta"], 75.0)

    def test_comparison_reports_per_participant(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(self.url)
        rows = {r["key"]: r for r in resp.context["by_user"]}
        self.assertEqual(rows["p1"]["mouse"], 3)
        self.assertEqual(rows["p2"]["keyboard"], 3)
        # The comparison table is research data: it identifies people by study
        # code, never by the address they log in with.
        self.assertNotIn("p1@study.local", rows)

    def test_shortcut_breakdown(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(self.url)
        self.assertEqual(dict(resp.context["shortcuts"]), {"Enter": 1, "ArrowRight": 3})

    def test_csv_export(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(reverse("admin:web_interactionevent_comparison_csv"))
        self.assertEqual(resp.status_code, 200)
        body = b"".join(resp.streaming_content).decode()
        self.assertIn("keyboard_share_pct", body)
        self.assertIn("p1", body)
        self.assertNotIn("@", body, "no email address may reach the export")

    def test_link_from_the_event_changelist(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(reverse("admin:web_interactionevent_changelist"))
        self.assertContains(resp, self.url)

    def test_empty_state_does_not_error(self):
        InteractionEvent.objects.all().delete()
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context["total"], 0)


class SignOutOnStudyPagesTests(StudyFlowTestCase):
    """Every participant-facing page needs a way out, not just the task."""

    LINK = 'class="hdr-signout"'

    def test_instructions_has_sign_out(self):
        self.login_participant()
        body = self.client.get(reverse("web:study_instructions")).content.decode()
        self.assertIn(self.LINK, body)
        self.assertIn(f'href="{reverse("web:logout")}"', body)

    def test_post_study_has_sign_out(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        body = self.client.get(reverse("web:study_post")).content.decode()
        self.assertIn(self.LINK, body)

    def test_post_study_warns_before_discarding_typed_answers(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        body = self.client.get(reverse("web:study_post")).content.decode()
        self.assertIn("confirm(", body)
        self.assertIn("will be lost", body)

    def test_done_page_has_sign_out(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        self.client.post(
            reverse("web:study_post"),
            {
                "tf1": "true", "tf2": "false", "tf3": "true", "tf4": "false",
                "motive": "a", "connections": "b", "timeline": "c", "threads": "d",
            },
        )
        body = self.client.get(reverse("web:study_done")).content.decode()
        self.assertIn(self.LINK, body)

    def test_sign_out_from_a_study_page_returns_to_the_study_login(self):
        self.login_participant()
        self.client.get(reverse("web:study_instructions"))
        resp = self.client.get(reverse("web:logout"))
        self.assertRedirects(
            resp, reverse("web:study_login"), fetch_redirect_response=False
        )
        self.assertNotIn("_auth_user_id", self.client.session)


class WiderInteractionCaptureTests(StudyFlowTestCase):
    def setUp(self):
        super().setUp()
        self.participant.study_model_id = "m_test"
        self.participant.save()
        self.url = reverse("web:study_events")

    def test_all_new_action_types_are_accepted(self):
        self.login_participant()
        actions = [
            "click", "dblclick", "contextmenu", "key", "shortcut", "search",
            "change", "submit", "copy", "paste", "select",
            "modal_open", "modal_close", "scroll", "resize", "focus", "blur", "page",
        ]
        resp = self.client.post(
            self.url,
            data=json.dumps({"events": [
                {"action": a, "modality": "other"} for a in actions
            ]}),
            content_type="application/json",
        )
        self.assertEqual(resp.json()["stored"], len(actions))
        self.assertEqual(
            set(InteractionEvent.objects.values_list("action", flat=True)), set(actions)
        )

    def test_collector_wires_every_documented_listener(self):
        body = read_template_source("study/_telemetry.html")
        for evt in ("dblclick", "contextmenu", "change", "input", "copy", "paste",
                    "mouseup", "shown.bs.modal", "hidden.bs.modal", "resize"):
            self.assertIn(f"'{evt}'", body, f"no listener for {evt}")

    def test_elements_can_opt_out_of_logging(self):
        body = read_template_source("study/_telemetry.html")
        self.assertIn("[data-no-log]", body)
        self.assertIn("suppressed(ev.target)", body)

    def test_modal_dwell_time_is_captured(self):
        self.login_participant()
        self.client.post(
            self.url,
            data=json.dumps({"events": [
                {"action": "modal_open", "modality": "other", "target": "#docDetailModal"},
                {"action": "modal_close", "modality": "other", "target": "#docDetailModal",
                 "detail": {"open_ms": 8400}},
            ]}),
            content_type="application/json",
        )
        closed = InteractionEvent.objects.get(action="modal_close")
        self.assertEqual(closed.detail["open_ms"], 8400)
        self.assertEqual(closed.target, "#docDetailModal")

    def test_search_queries_are_captured(self):
        self.login_participant()
        self.client.post(
            self.url,
            data=json.dumps({"events": [
                {"action": "search", "modality": "keyboard", "target": "#docSearchInput",
                 "detail": {"query": "off-target edits"}},
            ]}),
            content_type="application/json",
        )
        self.assertEqual(
            InteractionEvent.objects.get(action="search").detail["query"],
            "off-target edits",
        )


class StudyResponseExportTests(TestCase):
    """The outcome data had no export at all; this is the analysis file."""

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.group = StudyGroup.objects.create(name="Cond A", slug="cond-a")
        self.p1 = User.objects.create_user(
            email="p001@study.local", password="pw", study_model_id="m_alpha",
            study_group=self.group, participant_code="p001",
        )
        session = study_flow.get_or_create_session(self.p1)
        study_flow.record_consent(session)
        study_flow.begin_task(session, self.p1)
        study_flow.finish_task(session, early=True)
        StudyResponse.objects.create(
            user=self.p1,
            study_group=self.group,
            model_id="m_alpha",
            tf_answers={"tf1": "true", "tf2": "true", "tf3": "declined"},
            tf_score=1,
            tf_answered=2,
            open_answers={
                "motive": {"text": "The settlement, ended it.", "citations": ["d1", "d2"]},
                "connections": {"text": "", "citations": []},
                "timeline": {"text": "Eight months", "citations": []},
                "threads": {"text": "Separate", "citations": []},
            },
            task_duration_s=421,
            form_duration_ms=90000,
        )
        self.url = reverse("admin:web_studyresponse_responses_csv")

    def body(self, resp):
        return b"".join(resp.streaming_content).decode()

    def rows(self):
        import csv, io

        return list(csv.reader(io.StringIO(self.body(self.client.get(self.url)))))

    def test_one_row_per_participant(self):
        rows = self.rows()
        self.assertEqual(len(rows), 2, "header plus one response")

    def test_columns_cover_the_instrument_and_the_condition(self):
        header = self.rows()[0]
        for col in ("participant", "dashboard", "study_group", "active_learning",
                    "consent_version", "consented_at", "task_duration_s",
                    "finished_early", "tf1", "tf1_correct", "tf_score",
                    "tf_answered", "motive", "motive_citations", "submitted_at"):
            self.assertIn(col, header)

    def test_row_carries_the_values(self):
        header, row = self.rows()
        record = dict(zip(header, row))
        self.assertEqual(record["participant"], "p001")
        self.assertEqual(record["dashboard"], "m_alpha")
        self.assertEqual(record["study_group"], "Cond A")
        self.assertEqual(record["consent_version"], study_flow.CONSENT_VERSION)
        self.assertEqual(record["tf_score"], "1")
        self.assertEqual(record["tf_answered"], "2")
        self.assertEqual(record["finished_early"], "1")
        self.assertEqual(record["active_learning"], "1")
        self.assertEqual(record["tf1"], "true")
        self.assertEqual(record["tf1_correct"], "1")
        self.assertEqual(record["tf2_correct"], "0", "tf2's key is false")
        self.assertEqual(record["tf3"], "declined")
        self.assertEqual(record["motive_citations"], "d1 d2")
        self.assertEqual(record["connections"], "")

    def test_commas_and_unicode_survive_the_round_trip(self):
        """A free-text answer is the most likely thing to break a CSV."""
        response = StudyResponse.objects.get()
        response.open_answers["motive"]["text"] = 'He said "no", then — 🙂\nnew line'
        response.save()
        record = dict(zip(*self.rows()))
        self.assertEqual(record["motive"], 'He said "no", then — 🙂\nnew line')

    def test_export_carries_no_email_addresses(self):
        self.assertNotIn("@", self.body(self.client.get(self.url)))

    def test_empty_export_is_a_valid_header_only_file(self):
        StudyResponse.objects.all().delete()
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertIn("participant", rows[0])

    def test_a_response_without_a_session_does_not_break_the_export(self):
        """Defensive: the export must not 500 on an odd row."""
        ParticipantSession.objects.all().delete()
        record = dict(zip(*self.rows()))
        self.assertEqual(record["participant"], "p001")
        self.assertEqual(record["consent_version"], "")
        self.assertEqual(record["active_learning"], "")

    def test_download_link_is_on_the_changelist(self):
        resp = self.client.get(reverse("admin:web_studyresponse_changelist"))
        self.assertContains(resp, self.url)


class TaskMinutesTests(StudyFlowTestCase):
    """Exposure time is per-condition, and frozen once a clock is running."""

    def test_defaults_to_the_protocol_thirty(self):
        self.assertEqual(study_flow.task_minutes(self.participant), 30)

    def test_group_override_applies(self):
        self.group.task_minutes = 45
        self.group.save()
        self.participant.refresh_from_db()
        self.assertEqual(study_flow.task_minutes(self.participant), 45)

    def test_per_user_override_beats_the_group(self):
        self.group.task_minutes = 45
        self.group.save()
        self.participant.task_minutes = 10
        self.participant.save()
        self.participant.refresh_from_db()
        self.assertEqual(study_flow.task_minutes(self.participant), 10)

    def test_snapshotted_at_task_start(self):
        self.participant.task_minutes = 12
        self.participant.save()
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.task_minutes, 12)
        span = (session.task_ends_at - session.task_started_at).total_seconds()
        self.assertAlmostEqual(span, 12 * 60, delta=2)

    def test_changing_it_mid_run_does_not_move_a_running_clock(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        before = ParticipantSession.objects.get(pk=self.participant.pk).task_ends_at

        self.participant.task_minutes = 90
        self.participant.save()
        self.client.get(reverse("web:study_timer"))

        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.task_ends_at, before)
        self.assertEqual(session.task_minutes, 30)


class InstrumentVersionTests(StudyFlowTestCase):
    """Scores must stay interpretable if the questions are ever revised."""

    def submit(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        self.client.post(
            reverse("web:study_post"),
            {"tf1": "true", "tf2": "false", "tf3": "true", "tf4": "false",
             "motive": "a", "connections": "b", "timeline": "c", "threads": "d"},
        )

    def test_version_is_stamped_on_submission(self):
        self.submit()
        self.assertEqual(
            StudyResponse.objects.get().instrument_version,
            study_flow.INSTRUMENT_VERSION,
        )

    def test_study_flow_still_re_exports_the_instrument(self):
        """Call sites have always used study_flow.TF_KEY; keep that working."""
        from web import instruments

        self.assertIs(study_flow.TF_KEY, instruments.TF_KEY)
        self.assertIs(study_flow.OPEN_QUESTIONS, instruments.OPEN_QUESTIONS)
        self.assertIs(study_flow.score_tf, instruments.score_tf)

    def test_an_earlier_response_keeps_the_version_it_was_scored_under(self):
        self.submit()
        with mock.patch.object(study_flow, "INSTRUMENT_VERSION", "corpusbio-v2"):
            self.assertEqual(
                StudyResponse.objects.get().instrument_version, "corpusbio-v1"
            )


class PreflightAndProgressTests(TestCase):
    """Both pages must render when the backend is down -- that is the point."""

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.group = StudyGroup.objects.create(
            name="Cond A", slug="cond-a", default_model_id="m_group"
        )
        self.p1 = User.objects.create_user(
            email="p001@study.local", password="pw", study_model_id="m_alpha",
            participant_code="p001",
        )
        self.p2 = User.objects.create_user(
            email="p002@study.local", password="pw", study_group=self.group,
            participant_code="p002",
        )
        self.orphan = User.objects.create_user(
            email="p003@study.local", password="pw", participant_code="p003"
        )
        self.preflight = reverse("tova_admin_preflight")
        self.progress = reverse("tova_admin_progress")

    def test_preflight_renders_with_no_backend(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]), \
             mock.patch("web.views._study_corpus_size", side_effect=RuntimeError("down")):
            resp = self.client.get(self.preflight)
        self.assertEqual(resp.status_code, 200)

    def test_preflight_counts_direct_and_inherited_assignments(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]), \
             mock.patch("web.views._study_corpus_size", return_value=600):
            resp = self.client.get(self.preflight)
        rows = {r["model_id"]: r for r in resp.context["rows"]}
        self.assertEqual(rows["m_alpha"]["participants"], 1)
        self.assertEqual(rows["m_group"]["participants"], 1, "inherited from the group")

    def test_preflight_flags_a_model_missing_from_the_store(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]), \
             mock.patch("web.views._study_corpus_size", return_value=600):
            resp = self.client.get(self.preflight)
        rows = {r["model_id"]: r for r in resp.context["rows"]}
        self.assertFalse(rows["m_alpha"]["ready"])
        self.assertIn("model store", rows["m_alpha"]["error"])

    def test_preflight_marks_a_healthy_dashboard_ready(self):
        with mock.patch(
            "web.admin.available_dashboards",
            return_value=[("m_alpha", "Alpha  (m_alpha · corpus c1)")],
        ), mock.patch("web.views._study_corpus_size", return_value=600):
            resp = self.client.get(self.preflight)
        rows = {r["model_id"]: r for r in resp.context["rows"]}
        self.assertTrue(rows["m_alpha"]["ready"])
        self.assertEqual(rows["m_alpha"]["corpus_size"], 600)
        self.assertIsNone(rows["m_alpha"]["error"])

    def test_preflight_counts_people_who_cannot_start(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]), \
             mock.patch("web.views._study_corpus_size", return_value=1):
            resp = self.client.get(self.preflight)
        self.assertEqual(resp.context["unassigned"], 1, "the orphan")

    def test_progress_renders_empty(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(self.progress)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context["totals"]["total"], 0)

    def test_progress_counts_the_funnel(self):
        s1 = study_flow.get_or_create_session(self.p1)
        study_flow.record_consent(s1)
        study_flow.begin_task(s1, self.p1)
        study_flow.finish_task(s1, early=True)
        StudyResponse.objects.create(
            user=self.p1, model_id="m_alpha", tf_score=3, tf_answered=4,
            task_duration_s=400,
        )
        study_flow.complete_study(s1)
        study_flow.get_or_create_session(self.p2)  # consent not given

        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(self.progress)

        totals = resp.context["totals"]
        self.assertEqual(totals["total"], 2)
        self.assertEqual(totals["consented"], 1)
        self.assertEqual(totals["started"], 1)
        self.assertEqual(totals["complete"], 1)
        self.assertEqual(totals["submitted"], 1)

        rows = {r["model_id"]: r for r in resp.context["rows"]}
        self.assertEqual(rows["m_alpha"]["mean_score"], 3.0)
        self.assertEqual(rows["m_alpha"]["complete"], 1)

    def test_both_pages_are_staff_only(self):
        self.client.force_login(self.p1)
        for url in (self.preflight, self.progress):
            with self.subTest(url=url):
                resp = self.client.get(url)
                self.assertNotEqual(resp.status_code, 200)


class StudyEdgeCaseTests(StudyFlowTestCase):
    """Things that only break once someone tries them."""

    def test_next_parameter_cannot_be_used_to_skip_the_flow(self):
        """Signing in with ?next=/study/post must not bypass consent."""
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "pw",
             "next": reverse("web:study_post")},
            follow=True,
        )
        self.assertEqual(resp.request["PATH_INFO"], reverse("web:study_consent"))
        self.assertIsNone(
            ParticipantSession.objects.get(pk=self.participant.pk).consented_at
        )

    def test_telemetry_still_posts_while_at_the_consent_gate(self):
        """study_events has to be reachable at every stage or logging has holes."""
        self.login_participant(consent=False)
        resp = self.client.post(
            reverse("web:study_events"),
            data=json.dumps({"events": [
                {"action": "click", "modality": "mouse", "target": "consent_accept", "t": 10}
            ]}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["stored"], 1)
        self.assertEqual(InteractionEvent.objects.get().stage, "consent")

    def test_a_duplicate_study_code_does_not_crash_creation(self):
        """p001@study.local and p001@example.com both want "p001"."""
        from web.admin import participant_code_for

        User.objects.create_user(
            email="p042@study.local", password="pw", participant_code="p042"
        )
        self.assertIsNone(participant_code_for("p042@example.com"))
        # ...but the account that already owns it keeps it on a re-run.
        self.assertEqual(participant_code_for("p042@study.local"), "p042")
        # ...and a code nobody holds is still handed out.
        self.assertEqual(participant_code_for("p043@example.com"), "p043")

    def test_zero_task_minutes_falls_back_rather_than_ending_instantly(self):
        self.participant.task_minutes = 0
        self.participant.save()
        self.assertEqual(study_flow.task_minutes(self.participant), 30)

    def test_finishing_twice_does_not_move_the_stage_on_again(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        self.client.post(reverse("web:study_finish"))
        session = ParticipantSession.objects.get(pk=self.participant.pk)
        self.assertEqual(session.stage, ParticipantSession.STAGE_POST)

    def test_consent_page_is_not_reachable_once_signed_out(self):
        resp = self.client.get(reverse("web:study_consent"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])

    def test_deleting_a_participant_removes_their_research_data(self):
        """Withdrawal has to actually remove things."""
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        ResearcherNote.objects.create(user=self.participant, text="note")
        InteractionEvent.objects.create(
            user=self.participant, action="click", modality="mouse"
        )
        StudyResponse.objects.create(user=self.participant, model_id="m_test")

        self.participant.delete()

        self.assertFalse(ParticipantSession.objects.exists())
        self.assertFalse(StudyResponse.objects.exists())
        self.assertFalse(ResearcherNote.objects.exists())
        self.assertFalse(InteractionEvent.objects.exists())

    def test_unknown_event_actions_are_dropped_not_stored(self):
        self.login_participant()
        self.client.post(
            reverse("web:study_events"),
            data=json.dumps({"events": [
                {"action": "keylog", "modality": "keyboard", "target": "x"},
                {"action": "click", "modality": "mouse", "target": "y"},
            ]}),
            content_type="application/json",
        )
        self.assertEqual(
            list(InteractionEvent.objects.values_list("action", flat=True)), ["click"]
        )


class ModeratedSessionTests(TestCase):
    """The protocol has two administration modes; the data has to say which."""

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.p1 = User.objects.create_user(
            email="p001@study.local", password="pw", study_model_id="m_alpha",
            participant_code="p001",
        )
        self.session = study_flow.get_or_create_session(self.p1)

    def test_sessions_are_independent_by_default(self):
        self.assertFalse(self.session.moderated)

    def test_admin_action_marks_a_session_moderated(self):
        url = reverse("admin:web_participantsession_changelist")
        self.client.post(url, {
            "action": "mark_moderated",
            "_selected_action": [str(self.p1.pk)],
        })
        self.session.refresh_from_db()
        self.assertTrue(self.session.moderated)

    def test_admin_action_marks_it_back(self):
        ParticipantSession.objects.update(moderated=True)
        url = reverse("admin:web_participantsession_changelist")
        self.client.post(url, {
            "action": "mark_independent",
            "_selected_action": [str(self.p1.pk)],
        })
        self.session.refresh_from_db()
        self.assertFalse(self.session.moderated)

    def test_notes_are_linked_to_the_participant_and_the_author(self):
        note = ResearcherNote.objects.create(
            user=self.p1, author=self.admin, text="Kept re-reading the settlement thread."
        )
        self.assertEqual(note.user, self.p1)
        self.assertEqual(note.author, self.admin)
        self.assertIn("p001", str(note))

    def test_notes_survive_the_author_being_deleted(self):
        """Losing a staff account must not destroy the research record."""
        note = ResearcherNote.objects.create(
            user=self.p1, author=self.admin, text="Thought aloud about the timeline."
        )
        self.admin.delete()
        note.refresh_from_db()
        self.assertIsNone(note.author)
        self.assertEqual(note.text, "Thought aloud about the timeline.")

    def test_notes_are_writable_in_the_admin(self):
        """Unlike the other study tables, which are deliberately read-only."""
        resp = self.client.get(reverse("admin:web_researchernote_add"))
        self.assertEqual(resp.status_code, 200)

    def test_moderated_flag_reaches_the_response_export(self):
        self.session.moderated = True
        self.session.save()
        StudyResponse.objects.create(user=self.p1, model_id="m_alpha")
        resp = self.client.get(reverse("admin:web_studyresponse_responses_csv"))
        import csv, io

        rows = list(csv.reader(io.StringIO(b"".join(resp.streaming_content).decode())))
        record = dict(zip(*rows))
        self.assertEqual(record["moderated"], "1")


class InteractionCsvExportTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.group = StudyGroup.objects.create(name="Cond A", slug="cond-a")
        self.p1 = User.objects.create_user(
            email="p1@study.local", password="pw", study_model_id="m_alpha",
            study_group=self.group, participant_code="p1",
        )
        self.p2 = User.objects.create_user(
            email="p2@study.local", password="pw", study_model_id="m_beta",
            participant_code="p2",
        )
        InteractionEvent.objects.create(
            user=self.p1, study_group=self.group, model_id="m_alpha", stage="task",
            page="dashboard", action="search", modality="keyboard",
            target="#docSearchInput", client_ms=4200, detail={"query": "settlement"},
        )
        InteractionEvent.objects.create(
            user=self.p2, model_id="m_beta", stage="task", page="dashboard",
            action="click", modality="mouse", target="#themeChart", client_ms=900,
        )
        self.url = reverse("admin:web_interactionevent_events_csv")

    def body(self, resp):
        return b"".join(resp.streaming_content).decode()

    def test_full_export_has_a_row_per_event(self):
        body = self.body(self.client.get(self.url))
        lines = [ln for ln in body.strip().splitlines() if ln]
        self.assertEqual(len(lines), 3, "header plus two events")
        self.assertIn("p1", body)
        self.assertIn("p2", body)

    def test_export_carries_no_email_addresses(self):
        """The consent says data is linked only to a study-specific code."""
        body = self.body(self.client.get(self.url))
        self.assertNotIn("@", body)
        self.assertNotIn("study.local", body)

    def test_export_carries_the_columns_analysis_needs(self):
        header = self.body(self.client.get(self.url)).splitlines()[0]
        for col in ("created_at", "participant", "dashboard", "study_group", "stage",
                    "page", "action", "modality", "target", "client_ms", "detail"):
            self.assertIn(col, header)

    def test_detail_json_survives_the_round_trip(self):
        row = [ln for ln in self.body(self.client.get(self.url)).splitlines()
               if ",p1," in ln][0]
        self.assertIn("settlement", row)
        self.assertIn("Cond A", row)

    def test_selected_export_action(self):
        resp = self.client.post(
            reverse("admin:web_interactionevent_changelist"),
            {
                "action": "export_selected_csv",
                "_selected_action": [str(InteractionEvent.objects.first().pk)],
            },
        )
        body = self.body(resp)
        lines = [ln for ln in body.strip().splitlines() if ln]
        self.assertEqual(len(lines), 2, "header plus the single selected event")

    def test_download_link_is_on_the_changelist(self):
        with mock.patch("web.admin.available_dashboards", return_value=[]):
            resp = self.client.get(reverse("admin:web_interactionevent_changelist"))
        self.assertContains(resp, self.url)

    def test_export_is_staff_only(self):
        self.client.force_login(User.objects.create_user(email="joe@x.com", password="pw"))
        resp = self.client.get(self.url)
        self.assertIn(resp.status_code, (302, 403))


class LogSuppressionTests(StudyFlowTestCase):
    """Decorative chrome is muted; the controls inside it are not."""

    def collector(self):
        return read_template_source("study/_telemetry.html")

    def test_override_is_implemented(self):
        js = self.collector()
        self.assertIn("off.contains(on)", js)
        self.assertIn("[data-log]", js)

    def test_standalone_headers_are_muted(self):
        self.login_participant()
        body = self.client.get(reverse("web:study_instructions")).content.decode()
        self.assertIn("<header data-no-log>", body)

    def test_sign_out_stays_logged_inside_a_muted_header(self):
        self.login_participant()
        body = self.client.get(reverse("web:study_instructions")).content.decode()
        header = body[body.index("<header data-no-log>"):body.index("</header>")]
        self.assertIn('data-log="sign_out"', header)

    def test_navbar_brand_is_muted_but_nav_is_not(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            body = self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()
        self.assertIn('<a data-no-log class="navbar-brand', body)
        # The whole header must not be blanketed, or the timer controls go silent.
        self.assertNotIn('<header data-no-log class="dash-header"', body)

    def test_timer_controls_carry_their_own_log_hooks(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        with mock.patch("web.views._model_nav_context", return_value={"model_id": "m_test"}):
            body = self.client.get(
                reverse("web:dashboard"), {"model_id": "m_test"}
            ).content.decode()
        self.assertIn('data-log="finish_early"', body)
        self.assertIn('data-log="sign_out"', body)

    def test_post_study_header_is_muted(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        body = self.client.get(reverse("web:study_post")).content.decode()
        self.assertIn("<header data-no-log>", body)
        header = body[body.index("<header data-no-log>"):body.index("</header>")]
        self.assertIn('data-log="sign_out"', header)


class FlashMessageHygieneTests(StudyFlowTestCase):
    """Messages must not survive a sign-out.

    Study screens do not render the message block, so anything queued during a
    run stays unread. Session flush alone does not clear it -- Django's default
    FallbackStorage writes messages to a cookie.
    """

    def shown(self, response):
        from django.contrib.messages import get_messages

        return [str(m) for m in get_messages(response.wsgi_request)]

    def test_signing_in_queues_nothing_for_a_participant(self):
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "pw"},
            follow=True,
        )
        self.assertEqual(self.shown(resp), [])

    def test_sign_out_shows_only_the_sign_out_message(self):
        self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "pw"},
        )
        self.client.post(reverse("web:study_begin"))
        resp = self.client.get(reverse("web:logout"), follow=True)
        self.assertEqual(self.shown(resp), ["You have been signed out."])

    def test_stale_messages_are_dropped_at_sign_out(self):
        """Anything queued mid-run, from any source, must not leak out."""
        from django.contrib import messages as dj_messages

        self.login_participant()
        request = self.client.get(reverse("web:study_instructions")).wsgi_request
        dj_messages.warning(request, "left over from mid-run")
        request.session.save()

        resp = self.client.get(reverse("web:logout"), follow=True)
        body = resp.content.decode()
        self.assertNotIn("left over from mid-run", body)

    def test_login_errors_still_reach_the_participant(self):
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "wrong"},
            follow=True,
        )
        self.assertIn("Invalid email or password.", self.shown(resp))

    def test_staff_still_get_the_signed_in_confirmation(self):
        resp = self.client.post(
            reverse("web:login"),
            {"email": self.staff.email, "password": "pw"},
            follow=True,
        )
        self.assertIn("Signed in successfully.", self.shown(resp))


class ModelAccessForParticipantsTests(TestCase):
    """Participants own nothing; access follows the dashboard they were assigned.

    Regression: the study check used to sit after the ownership branch, which
    returns _verify_corpus_ownership() outright whenever the model has a corpus.
    Every participant was therefore refused their own dashboard.
    """

    MINE = "m_mine"
    THEIRS = "m_theirs"

    def setUp(self):
        self.participant = User.objects.create_user(
            email="p@s.local", password="pw", study_model_id=self.MINE
        )
        self.outsider = User.objects.create_user(email="nobody@x.com", password="pw")

    def fake_api(self, corpus_id="c_mine"):
        """Model exists, has a corpus, and is owned by nobody in this test."""
        resp = mock.Mock(status_code=200)
        resp.json = lambda: {"id": self.MINE, "owner_id": "someone-else",
                             "corpus_id": corpus_id, "metadata": {}}
        # runtime.py imports requests inside each function, so there is no
        # module attribute to patch -- patch it at the source.
        return mock.patch("requests.get", return_value=resp)

    def test_participant_can_reach_their_assigned_model(self):
        with self.fake_api():
            self.assertTrue(
                R._verify_model_ownership(self.MINE, str(self.participant.pk))
            )

    def test_participant_cannot_reach_another_condition(self):
        with self.fake_api():
            self.assertFalse(
                R._verify_model_ownership(self.THEIRS, str(self.participant.pk))
            )

    def test_ownership_still_applies_to_everyone_else(self):
        with self.fake_api():
            self.assertFalse(
                R._verify_model_ownership(self.MINE, str(self.outsider.pk))
            )

    def test_corpus_behind_the_assigned_model_is_readable(self):
        R._CACHE_MODEL_INFO.clear()
        with self.fake_api(corpus_id="c_mine"):
            self.assertTrue(
                R._verify_corpus_ownership("c_mine", str(self.participant.pk))
            )

    def test_other_corpora_are_not(self):
        R._CACHE_MODEL_INFO.clear()
        with self.fake_api(corpus_id="c_mine"):
            self.assertFalse(
                R._verify_corpus_ownership("c_other", str(self.participant.pk))
            )

    def test_a_participant_with_no_dashboard_gets_nothing(self):
        stray = User.objects.create_user(email="stray@x.com", password="pw")
        stray.study_group = StudyGroup.objects.create(name="G", slug="g")
        stray.save()
        with self.fake_api():
            self.assertFalse(R._verify_model_ownership(self.MINE, str(stray.pk)))

    def test_unauthenticated_is_refused(self):
        self.assertFalse(R._verify_model_ownership(self.MINE, ""))


class ActiveLearningKeyboardTests(TestCase):
    """Keyboard affordances on the AL page.

    Keyboard-vs-mouse is a measured outcome of this study, so a shortcut that
    is mislabelled or invisible to assistive tech biases the very thing being
    measured -- it turns a preference question into a discoverability question.
    """

    def source(self):
        return read_template_source("active_learning_two.html")

    def test_the_hint_matches_what_the_handler_accepts(self):
        src = self.source()
        # The handler takes 1-9; the hint used to advertise 1-3.
        self.assertIn('e.key >= "1" && e.key <= "9"', src)
        self.assertIn("<kbd>1</kbd>–<kbd>9</kbd> pick", src)

    def test_the_shortcut_hint_is_not_hidden_from_assistive_tech(self):
        src = self.source()
        hint = src[src.index('class="al2-kbd-hint"'):][:200]
        self.assertNotIn("aria-hidden", hint)

    def test_suggestion_count_is_not_hardcoded(self):
        """It announced "of 3" however many suggestions came back."""
        src = self.source()
        self.assertNotIn("of 3 (press", src)
        self.assertIn("of ${total}", src)


class ActiveLearningPrototypeTests(StudyFlowTestCase):
    """The prototype is off unless deliberately enabled."""

    def setUp(self):
        super().setUp()
        self.participant.study_model_id = "m_test"
        self.participant.save()
        self.url = reverse("web:active_learning_three")

    def test_404s_when_the_flag_is_off(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        resp = self.client.get(self.url, {"model_id": "m_test"})
        self.assertEqual(resp.status_code, 404)

    @override_settings(TOVA_ENABLE_AL3=True)
    def test_renders_when_enabled(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        with mock.patch("web.services.runtime._verify_model_ownership", return_value=True):
            resp = self.client.get(self.url, {"model_id": "m_test"})
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn('id="al3"', body)
        self.assertIn("css/al3.css", body)
        self.assertIn("js/al3.js", body)

    @override_settings(TOVA_ENABLE_AL3=True)
    def test_style_and_behaviour_are_not_inlined(self):
        """Half the proposal is that the page stops being one 3,000-line file."""
        src = read_template_source("active_learning_three.html")
        self.assertNotIn("<style", src)
        self.assertNotIn("addEventListener", src)

    @override_settings(TOVA_ENABLE_AL3=True)
    def test_blocked_for_a_participant_whose_condition_excludes_it(self):
        self.group.enable_active_learning = False
        self.group.save()
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        resp = self.client.get(self.url, {"model_id": "m_test"})
        self.assertEqual(resp.status_code, 302)

    @override_settings(TOVA_ENABLE_AL3=True)
    def test_is_not_reachable_before_the_task_begins(self):
        self.login_participant()
        resp = self.client.get(self.url, {"model_id": "m_test"})
        self.assertEqual(resp.status_code, 302)
        self.assertIn(reverse("web:study_instructions"), resp["Location"])


class LabelEventTests(StudyFlowTestCase):
    """The annotation record: labels, and how the participant arrived at them.

    Document labels are written through to the backend, so before this they
    existed nowhere in the study database -- the central artifact of the
    annotation task was outside the analytic dataset.
    """

    URL = "web:active_learning_two_label"

    def setUp(self):
        super().setUp()
        self.participant.study_model_id = "m_test"
        self.participant.save()
        self.login_participant()
        self.client.post(reverse("web:study_begin"))

    def label(self, **payload):
        body = {"model_id": "m_test", "doc_id": "d1", "label": "Approval oversight"}
        body.update(payload)
        with mock.patch("web.services.runtime._verify_model_ownership", return_value=True), \
             mock.patch("web.views.requests.post") as post:
            post.return_value = mock.Mock(
                ok=True, status_code=200, content=b"{}",
                headers={"Content-Type": "application/json"},
            )
            return self.client.post(
                reverse(self.URL), data=json.dumps(body),
                content_type="application/json",
            )

    def test_a_label_is_recorded_in_the_study_database(self):
        self.label()
        e = LabelEvent.objects.get()
        self.assertEqual(e.user, self.participant)
        self.assertEqual(e.doc_id, "d1")
        self.assertEqual(e.label, "Approval oversight")
        self.assertEqual(e.model_id, "m_test")
        self.assertEqual(e.action, LabelEvent.APPLIED)

    def test_accepting_a_suggestion_is_distinguishable_from_typing(self):
        """The measure of whether the assistance is working at all."""
        self.label(doc_id="d1", provenance="suggestion", suggestion_rank=1)
        self.label(doc_id="d2", provenance="typed", label="Something else")

        a, b = LabelEvent.objects.order_by("sequence_index")
        self.assertEqual(a.provenance, LabelEvent.FROM_SUGGESTION)
        self.assertEqual(a.suggestion_rank, 1)
        self.assertEqual(b.provenance, LabelEvent.FROM_TYPED)
        self.assertIsNone(b.suggestion_rank)

    def test_agreement_with_the_model_is_computed(self):
        self.label(doc_id="d1", label="Approval oversight", model_label="Approval oversight")
        self.label(doc_id="d2", label="Something else", model_label="Approval oversight")
        a, b = LabelEvent.objects.order_by("sequence_index")
        self.assertTrue(a.agreed_with_model)
        self.assertFalse(b.agreed_with_model)

    def test_agreement_is_null_when_the_model_had_no_label(self):
        """Unknown must not masquerade as disagreement."""
        self.label(model_label="")
        self.assertIsNone(LabelEvent.objects.get().agreed_with_model)

    def test_confidence_and_dwell_are_captured_for_convergence_and_efficiency(self):
        self.label(model_confidence=0.61, dwell_ms=8400, modality="keyboard")
        e = LabelEvent.objects.get()
        self.assertAlmostEqual(e.model_confidence, 0.61)
        self.assertEqual(e.dwell_ms, 8400)
        self.assertEqual(e.modality, "keyboard")

    def test_sequence_index_orders_a_session(self):
        for i in range(3):
            self.label(doc_id=f"d{i}")
        self.assertEqual(
            list(LabelEvent.objects.order_by("sequence_index").values_list("sequence_index", flat=True)),
            [0, 1, 2],
        )

    def test_relabelling_the_same_document_is_marked_a_revision(self):
        self.label(doc_id="d1", label="First")
        self.label(doc_id="d1", label="Second")
        a, b = LabelEvent.objects.order_by("sequence_index")
        self.assertFalse(a.is_revision)
        self.assertTrue(b.is_revision)
        self.assertEqual(b.previous_label, "First")

    def test_removing_a_label_is_recorded_too(self):
        """An undo is a decision, not a gap in the sequence."""
        self.label(doc_id="d1")
        with mock.patch("web.services.runtime._verify_model_ownership", return_value=True), \
             mock.patch("web.views.requests.delete") as delete:
            delete.return_value = mock.Mock(
                ok=True, status_code=200, content=b"{}",
                headers={"Content-Type": "application/json"},
            )
            self.client.post(
                reverse("web:active_learning_two_unlabel"),
                data=json.dumps({"model_id": "m_test", "doc_id": "d1"}),
                content_type="application/json",
            )
        self.assertEqual(
            list(LabelEvent.objects.order_by("sequence_index").values_list("action", flat=True)),
            [LabelEvent.APPLIED, LabelEvent.REMOVED],
        )

    def test_nothing_is_recorded_when_upstream_rejects_the_label(self):
        """The study data must not claim a label the model never received."""
        with mock.patch("web.services.runtime._verify_model_ownership", return_value=True), \
             mock.patch("web.views.requests.post") as post:
            post.return_value = mock.Mock(
                ok=False, status_code=502, content=b"{}",
                headers={"Content-Type": "application/json"},
            )
            self.client.post(
                reverse(self.URL),
                data=json.dumps({"model_id": "m_test", "doc_id": "d1", "label": "x"}),
                content_type="application/json",
            )
        self.assertFalse(LabelEvent.objects.exists())

    def test_a_client_that_sends_no_provenance_still_records_the_label(self):
        """Enrichment is optional; the fact of the label is not."""
        self.label()
        e = LabelEvent.objects.get()
        self.assertEqual(e.provenance, LabelEvent.FROM_UNKNOWN)
        self.assertIsNone(e.suggestion_rank)
        self.assertEqual(e.label, "Approval oversight")

    def test_garbage_provenance_is_ignored_rather_than_stored(self):
        self.label(provenance="../etc/passwd", suggestion_rank="banana",
                   model_confidence="NaN-ish", dwell_ms=-5)
        e = LabelEvent.objects.get()
        self.assertEqual(e.provenance, LabelEvent.FROM_UNKNOWN)
        self.assertIsNone(e.suggestion_rank)
        self.assertIsNone(e.model_confidence)
        self.assertIsNone(e.dwell_ms)

    def test_labels_are_attributable_for_inter_annotator_agreement(self):
        """Two participants on one dashboard labelling one document."""
        other = User.objects.create_user(
            email="p002@study.local", password="pw",
            study_model_id="m_test", participant_code="p002",
        )
        self.label(doc_id="shared", label="Approval oversight")
        self.client.force_login(other)
        consent_for(other)
        self.client.post(reverse("web:study_begin"))
        self.label(doc_id="shared", label="Assay anomalies")

        rows = LabelEvent.objects.filter(doc_id="shared")
        self.assertEqual(rows.count(), 2)
        self.assertEqual(
            {r.user.export_id: r.label for r in rows},
            {"p001": "Approval oversight", "p002": "Assay anomalies"},
        )


class LabelEventExportTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.p = User.objects.create_user(
            email="p001@study.local", password="pw", participant_code="p001"
        )
        LabelEvent.objects.create(
            user=self.p, model_id="m_alpha", doc_id="d1", label="Approval oversight",
            provenance=LabelEvent.FROM_SUGGESTION, suggestion_rank=2,
            suggestions_shown=["Approval oversight", "Assay anomalies"],
            model_label="Assay anomalies", model_confidence=0.61,
            agreed_with_model=False, dwell_ms=8400, modality="keyboard",
            sequence_index=0,
        )
        self.url = reverse("admin:web_labelevent_labels_csv")

    def rows(self):
        import csv, io
        body = b"".join(self.client.get(self.url).streaming_content).decode()
        return list(csv.reader(io.StringIO(body)))

    def test_export_carries_what_each_question_needs(self):
        header, row = self.rows()
        rec = dict(zip(header, row))
        self.assertEqual(rec["participant"], "p001")
        self.assertEqual(rec["provenance"], "suggestion")
        self.assertEqual(rec["suggestion_rank"], "2")
        self.assertEqual(rec["model_confidence"], "0.61")
        self.assertEqual(rec["agreed_with_model"], "0")
        self.assertEqual(rec["dwell_ms"], "8400")
        self.assertEqual(rec["sequence_index"], "0")

    def test_export_carries_no_email(self):
        body = b"".join(self.client.get(self.url).streaming_content).decode()
        self.assertNotIn("@", body)

    def test_download_link_is_on_the_changelist(self):
        resp = self.client.get(reverse("admin:web_labelevent_changelist"))
        self.assertContains(resp, self.url)


class CorpusConsistencyTests(TestCase):
    """The backbone is the variable; the corpus must not be.

    If two conditions are built on different corpora, a difference between them
    cannot be attributed to the model. Caught before recruiting rather than
    during analysis.
    """

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        User.objects.create_user(email="p001@study.local", password="pw",
                                 study_model_id="m_alpha", participant_code="p001")
        User.objects.create_user(email="p002@study.local", password="pw",
                                 study_model_id="m_beta", participant_code="p002")
        self.url = reverse("tova_admin_preflight")

    def render(self, corpora):
        dashboards = [(m, f"{m}  ({m} · corpus {c})") for m, c in corpora.items()]
        with mock.patch("web.admin.available_dashboards", return_value=dashboards), \
             mock.patch("web.admin.corpus_for_model", side_effect=lambda m: corpora.get(m, "")), \
             mock.patch("web.views._study_corpus_size", return_value=600):
            return self.client.get(self.url)

    def test_same_corpus_is_not_flagged(self):
        resp = self.render({"m_alpha": "c_one", "m_beta": "c_one"})
        self.assertEqual(resp.context["mixed_corpora"], [])
        self.assertNotContains(resp, "not on the same corpus")

    def test_different_corpora_are_flagged_loudly(self):
        resp = self.render({"m_alpha": "c_one", "m_beta": "c_two"})
        self.assertEqual(resp.context["mixed_corpora"], ["c_one", "c_two"])
        self.assertContains(resp, "not on the same corpus")

    def test_an_unassigned_dashboard_does_not_trigger_the_warning(self):
        """Only conditions people are actually in can confound the comparison."""
        resp = self.render({"m_alpha": "c_one", "m_beta": "c_one", "m_spare": "c_other"})
        self.assertEqual(resp.context["mixed_corpora"], [])

    def test_the_corpus_is_shown_per_dashboard(self):
        resp = self.render({"m_alpha": "c_one", "m_beta": "c_one"})
        rows = {r["model_id"]: r for r in resp.context["rows"]}
        self.assertEqual(rows["m_alpha"]["corpus_id"], "c_one")


class OneDashboardPerParticipantTests(TestCase):
    """The multi-select divides a cohort; it never gives anyone two dashboards.

    The form lets an admin tick several dashboards, which reads as "each of
    these people gets all of these". It does not: participants are dealt out
    round-robin, one each. study_model_id is a single field, so the data model
    could not express two even if the form tried.
    """

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.url = reverse("admin:web_user_create_participants")
        self.dashboards = [("m_a", "A (m_a)"), ("m_b", "B (m_b)"), ("m_c", "C (m_c)")]

    def create(self, count, picked):
        with mock.patch("web.admin.available_dashboards", return_value=self.dashboards):
            return self.client.post(self.url, {
                "count": count, "emails": "", "prefix": "p",
                "domain": "study.local", "start": 1,
                "dashboards": picked, "active_learning": "on",
            })

    def participants(self):
        return User.objects.filter(is_staff=False).order_by("email")

    def test_each_participant_holds_exactly_one_dashboard(self):
        self.create(9, ["m_a", "m_b", "m_c"])
        for u in self.participants():
            with self.subTest(user=u.email):
                self.assertIn(u.study_model_id, {"m_a", "m_b", "m_c"})
                # A single CharField -- there is no second slot to fill.
                self.assertNotIn(",", u.study_model_id)

    def test_a_cohort_is_divided_evenly_not_duplicated(self):
        self.create(9, ["m_a", "m_b", "m_c"])
        counts = {}
        for u in self.participants():
            counts[u.study_model_id] = counts.get(u.study_model_id, 0) + 1
        self.assertEqual(counts, {"m_a": 3, "m_b": 3, "m_c": 3})
        self.assertEqual(self.participants().count(), 9, "9 people, not 27")

    def test_an_uneven_split_deals_the_remainder_round_robin(self):
        self.create(7, ["m_a", "m_b"])
        counts = {}
        for u in self.participants():
            counts[u.study_model_id] = counts.get(u.study_model_id, 0) + 1
        self.assertEqual(sorted(counts.values()), [3, 4])

    def test_ticking_one_dashboard_puts_everyone_in_one_condition(self):
        self.create(5, ["m_b"])
        self.assertEqual(
            set(self.participants().values_list("study_model_id", flat=True)), {"m_b"}
        )

    def test_the_form_says_so_in_words(self):
        with mock.patch("web.admin.available_dashboards", return_value=self.dashboards):
            body = self.client.get(self.url).content.decode()
        self.assertIn("exactly ONE dashboard", body)
        self.assertIn("Split them across", body)
        self.assertNotIn("What they get", body)


class ConditionGroupTests(TestCase):
    """A condition is (dashboard x active learning), and it has a name."""

    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.url = reverse("admin:web_user_create_participants")
        self.dashboards = [
            ("m_a", "LDA baseline  (m_a · corpus c_1)"),
            ("m_b", "CTM  (m_b · corpus c_1)"),
        ]

    def create(self, count, picked, al):
        with mock.patch("web.admin.available_dashboards", return_value=self.dashboards):
            return self.client.post(self.url, {
                "count": count, "emails": "", "prefix": "p",
                "domain": "study.local", "start": 1,
                "dashboards": picked, "active_learning": al,
            })

    def participants(self):
        return User.objects.filter(is_staff=False)

    def test_splitting_doubles_the_arms(self):
        self.create(8, ["m_a", "m_b"], "split")
        self.assertEqual(StudyGroup.objects.count(), 4)
        self.assertEqual(
            set(StudyGroup.objects.values_list("name", flat=True)),
            {"LDA baseline · AL on", "LDA baseline · AL off",
             "CTM · AL on", "CTM · AL off"},
        )

    def test_the_cohort_divides_evenly_between_conditions(self):
        self.create(8, ["m_a", "m_b"], "split")
        counts = {}
        for u in self.participants():
            counts[u.study_group.name] = counts.get(u.study_group.name, 0) + 1
        self.assertEqual(sorted(counts.values()), [2, 2, 2, 2])

    def test_a_group_carries_its_condition(self):
        self.create(4, ["m_a"], "split")
        on = StudyGroup.objects.get(name="LDA baseline · AL on")
        off = StudyGroup.objects.get(name="LDA baseline · AL off")
        self.assertTrue(on.enable_active_learning)
        self.assertFalse(off.enable_active_learning)
        self.assertEqual(on.default_model_id, "m_a")
        self.assertEqual(off.default_model_id, "m_a")

    def test_participants_inherit_al_from_their_group(self):
        """One place holds the answer to "was AL on for this person"."""
        self.create(4, ["m_a"], "split")
        for u in self.participants():
            with self.subTest(user=u.email):
                self.assertIsNone(u.enable_active_learning, "must inherit, not duplicate")
                expected = u.study_group.enable_active_learning
                self.assertEqual(
                    user_has_feature(u, FEATURE_ACTIVE_LEARNING), expected
                )

    def test_on_for_everyone_makes_one_arm_per_dashboard(self):
        self.create(4, ["m_a", "m_b"], "on")
        self.assertEqual(StudyGroup.objects.count(), 2)
        self.assertTrue(all(g.enable_active_learning for g in StudyGroup.objects.all()))

    def test_off_for_everyone_makes_one_arm_per_dashboard(self):
        self.create(4, ["m_a", "m_b"], "off")
        self.assertEqual(StudyGroup.objects.count(), 2)
        self.assertFalse(any(g.enable_active_learning for g in StudyGroup.objects.all()))
        for u in self.participants():
            self.assertFalse(user_has_feature(u, FEATURE_ACTIVE_LEARNING))

    def test_a_second_cohort_reuses_the_same_groups(self):
        self.create(4, ["m_a"], "split")
        with mock.patch("web.admin.available_dashboards", return_value=self.dashboards):
            self.client.post(self.url, {
                "count": 4, "emails": "", "prefix": "q",
                "domain": "study.local", "start": 1,
                "dashboards": ["m_a"], "active_learning": "split",
            })
        self.assertEqual(StudyGroup.objects.count(), 2, "no duplicate conditions")
        self.assertEqual(self.participants().count(), 8)

    def test_each_participant_still_holds_exactly_one_dashboard(self):
        self.create(8, ["m_a", "m_b"], "split")
        for u in self.participants():
            self.assertIn(u.study_model_id, {"m_a", "m_b"})
            self.assertEqual(u.study_model_id, u.study_group.default_model_id)


class ResumeAfterSignOutTests(StudyFlowTestCase):
    """What happens when a participant signs out and comes back.

    The ParticipantSession is a database row keyed on the user, so signing out
    flushes the auth cookie and nothing else. The stage survives. The clock,
    however, is an absolute deadline -- it does not pause.
    """

    def sign_in(self):
        """Sign in and report the immediate redirect target."""
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "pw"},
        )
        return resp["Location"]

    def sign_in_and_follow(self):
        """Where a browser actually ends up.

        Login redirects to the canonical stage URL; /study/task then forwards to
        the dashboard, and an expired task is closed by the middleware on that
        next request rather than at login. So the landing page takes more than
        one hop and only following the chain reflects what a participant sees.
        """
        resp = self.client.post(
            reverse("web:study_login"),
            {"email": self.participant.email, "password": "pw"},
            follow=True,
        )
        return resp.request["PATH_INFO"]

    def session(self):
        return ParticipantSession.objects.get(pk=self.participant.pk)

    def test_consent_is_not_asked_for_twice(self):
        self.login_participant()          # consents
        self.client.get(reverse("web:logout"))
        self.assertEqual(self.sign_in(), reverse("web:study_instructions"))
        self.assertIsNotNone(self.session().consented_at)

    def test_signing_out_before_starting_loses_nothing(self):
        self.login_participant()
        self.client.get(reverse("web:logout"))

        self.assertEqual(self.sign_in(), reverse("web:study_instructions"))
        self.client.post(reverse("web:study_begin"))
        s = self.session()
        self.assertEqual(s.stage, ParticipantSession.STAGE_TASK)
        self.assertGreater(s.seconds_remaining, 29 * 60, "full time still ahead")

    def test_returning_mid_task_resumes_with_the_time_that_is_left(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        started = self.session().task_ends_at
        self.client.get(reverse("web:logout"))

        # Ten minutes pass while they are signed out.
        s = self.session()
        s.task_started_at -= dt.timedelta(minutes=10)
        s.task_ends_at -= dt.timedelta(minutes=10)
        s.save()

        self.assertEqual(self.sign_in(), reverse("web:study_task"))
        back = self.session()
        self.assertEqual(back.stage, ParticipantSession.STAGE_TASK)
        self.assertLess(back.seconds_remaining, 21 * 60,
                        "the clock ran while they were away")
        self.assertGreater(back.seconds_remaining, 19 * 60)
        self.assertNotEqual(back.task_ends_at, started)

    def test_the_clock_does_not_pause_while_signed_out(self):
        """Signing out is not a way to buy more exploration time."""
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.get(reverse("web:logout"))

        s = self.session()
        s.task_started_at -= dt.timedelta(minutes=45)
        s.task_ends_at -= dt.timedelta(minutes=45)
        s.save()

        # Coming back after the deadline: login still points at the task, and
        # the middleware closes it on that request. The participant lands on the
        # questionnaire and never gets to explore again.
        self.assertEqual(self.sign_in_and_follow(), reverse("web:study_post"))
        self.assertEqual(self.session().stage, ParticipantSession.STAGE_POST)
        self.assertFalse(self.session().finished_early, "expired, not finished early")

    def test_returning_after_submitting_lands_on_the_thank_you_page(self):
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        self.client.post(reverse("web:study_post"), {
            "tf1": "true", "tf2": "false", "tf3": "true", "tf4": "false",
            "motive": "a", "connections": "b", "timeline": "c", "threads": "d",
        })
        self.client.get(reverse("web:logout"))
        self.assertEqual(self.sign_in(), reverse("web:study_done"))

    def test_answers_typed_but_not_submitted_are_lost(self):
        """Documents a real risk: the questionnaire is not saved as you type."""
        self.login_participant()
        self.client.post(reverse("web:study_begin"))
        self.client.post(reverse("web:study_finish"))
        self.client.get(reverse("web:study_post"))   # opens the form, types nothing
        self.client.get(reverse("web:logout"))

        self.assertEqual(self.sign_in(), reverse("web:study_post"))
        self.assertFalse(StudyResponse.objects.exists())
