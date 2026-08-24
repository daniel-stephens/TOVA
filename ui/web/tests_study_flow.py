"""Tests for the participant study flow: stage gating, timer, and submission."""

from __future__ import annotations

import datetime as dt
import json
from unittest import mock

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from web.models import (
    InteractionEvent,
    ParticipantSession,
    StudyGroup,
    StudyResponse,
    TopicRename,
    User,
)
from web import study_flow
from web.services import runtime as R


class StudyFlowTestCase(TestCase):
    def setUp(self):
        self.group = StudyGroup.objects.create(
            name="Condition A",
            slug="condition-a",
            default_model_id="m_test",
            enable_active_learning=True,
        )
        self.participant = User.objects.create_user(
            email="p001@study.local", password="pw", study_group=self.group
        )
        self.outsider = User.objects.create_user(email="nobody@example.com", password="pw")
        self.staff = User.objects.create_user(
            email="admin@example.com", password="pw", is_staff=True
        )

    def login_participant(self):
        self.client.force_login(self.participant)


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

    def test_missing_answers_are_rejected_and_input_preserved(self):
        self.advance_to_post()
        resp = self.client.post(
            reverse("web:study_post"), self.valid_payload(tf1="", motive="")
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(StudyResponse.objects.exists())
        body = resp.content.decode()
        self.assertIn("Marcos and Priya saw the same approval pattern.", body)

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
        self.client.force_login(user or self.participant)
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
            "active_learning": "default",
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
        body = open("../templates/study/_telemetry.html", encoding="utf-8").read()
        branch = body[body.index("} else if (key.length === 1) {"):]
        branch = branch[: branch.index("} else {")]
        self.assertIn("label = 'char';", branch)
        self.assertNotIn("label = key", branch)


class KeyboardMouseComparisonTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.p1 = User.objects.create_user(
            email="p1@s.local", password="pw", study_model_id="m_alpha"
        )
        self.p2 = User.objects.create_user(
            email="p2@s.local", password="pw", study_model_id="m_beta"
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
        self.assertEqual(rows["p1@s.local"]["mouse"], 3)
        self.assertEqual(rows["p2@s.local"]["keyboard"], 3)

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
        self.assertIn("p1@s.local", body)

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
        body = open("../templates/study/_telemetry.html", encoding="utf-8").read()
        for evt in ("dblclick", "contextmenu", "change", "input", "copy", "paste",
                    "mouseup", "shown.bs.modal", "hidden.bs.modal", "resize"):
            self.assertIn(f"'{evt}'", body, f"no listener for {evt}")

    def test_elements_can_opt_out_of_logging(self):
        body = open("../templates/study/_telemetry.html", encoding="utf-8").read()
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


class InteractionCsvExportTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser(email="root@example.com", password="pw")
        self.client.force_login(self.admin)
        self.group = StudyGroup.objects.create(name="Cond A", slug="cond-a")
        self.p1 = User.objects.create_user(
            email="p1@s.local", password="pw", study_model_id="m_alpha",
            study_group=self.group,
        )
        self.p2 = User.objects.create_user(
            email="p2@s.local", password="pw", study_model_id="m_beta"
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
        self.assertIn("p1@s.local", body)
        self.assertIn("p2@s.local", body)

    def test_export_carries_the_columns_analysis_needs(self):
        header = self.body(self.client.get(self.url)).splitlines()[0]
        for col in ("created_at", "participant", "dashboard", "study_group", "stage",
                    "page", "action", "modality", "target", "client_ms", "detail"):
            self.assertIn(col, header)

    def test_detail_json_survives_the_round_trip(self):
        row = [ln for ln in self.body(self.client.get(self.url)).splitlines()
               if "p1@s.local" in ln][0]
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
        return open("../templates/study/_telemetry.html", encoding="utf-8").read()

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
