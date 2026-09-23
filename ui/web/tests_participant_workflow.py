"""Participant navigation through the real labeling-to-questionnaire boundary."""

import datetime as dt
from html.parser import HTMLParser
from unittest import mock

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from web import study_flow
from web.models import LabelEvent, ParticipantSession, StudyResponse, User


class FinishForms(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.actions = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and "data-study-finish" in attrs:
            self.actions.append((attrs.get("method"), attrs.get("action")))


class ParticipantWorkflowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="workflow@study.local", password="test-password",
            study_model_id="workflow-model", participant_code="workflow",
        )
        self.client.force_login(self.user)
        self.nav = mock.patch(
            "web.views._model_nav_context", return_value={"model_id": "workflow-model"}
        )
        self.nav.start()
        self.addCleanup(self.nav.stop)
        self.size = mock.patch("web.views._study_corpus_size", return_value=2)
        self.size.start()
        self.addCleanup(self.size.stop)

    def enter_task(self):
        self.client.post(reverse("web:study_consent_accept"))
        self.client.get(reverse("web:study_instructions"))
        self.client.get(reverse("web:study_tutorial"))
        self.client.post(reverse("web:study_begin"))

    def test_participant_can_label_finish_and_submit_the_questionnaire(self):
        # The questionnaire is unavailable until exploration has ended.
        self.assertRedirects(
            self.client.get(reverse("web:study_post")),
            reverse("web:study_consent"), fetch_redirect_response=False,
        )
        self.enter_task()
        page = self.client.get(
            reverse("web:active_learning_two"), {"model_id": "workflow-model"}
        )
        self.assertContains(page, "Finish labeling")
        forms = FinishForms(page.content.decode()).actions
        self.assertGreaterEqual(len(forms), 2)
        for method, action in forms:
            self.assertEqual(method, "post")
            self.assertEqual(action, reverse("web:study_finish"))

        upstream = mock.Mock(
            ok=True, status_code=200,
            content=b'{"current_labels":{"doc-1":"Evidence"}}',
            headers={"Content-Type": "application/json"},
        )
        with mock.patch("web.views.requests.post", return_value=upstream) as label_request:
            response = self.client.post(
                reverse("web:active_learning_two_label"),
                {"model_id": "workflow-model", "doc_id": "doc-1", "label": "Evidence",
                 "user_id": "someone-else"},
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(label_request.call_args.kwargs["json"]["user_id"], str(self.user.pk))
        self.assertTrue(LabelEvent.objects.filter(user=self.user, label="Evidence").exists())

        # Submit the rendered form action as a browser without JavaScript would.
        response = self.client.post(forms[-1][1])
        self.assertRedirects(response, reverse("web:study_post"))
        session = ParticipantSession.objects.get(user=self.user)
        self.assertEqual(session.stage, ParticipantSession.STAGE_POST)
        self.assertTrue(session.finished_early)
        self.assertRedirects(
            self.client.get(reverse("web:active_learning_two"), {"model_id": "workflow-model"}),
            reverse("web:study_post"), fetch_redirect_response=False,
        )
        self.assertRedirects(
            self.client.post(reverse("web:study_post"), {}), reverse("web:study_done")
        )
        self.assertTrue(StudyResponse.objects.filter(user=self.user).exists())

    def test_timer_expiry_hands_off_to_questionnaire_without_early_finish(self):
        self.enter_task()
        now = timezone.now()
        session = ParticipantSession.objects.get(user=self.user)
        session.task_elapsed_s = session.task_total_seconds - 1
        session.last_seen_at = now - dt.timedelta(seconds=2)
        session.save(update_fields=["task_elapsed_s", "last_seen_at"])
        with mock.patch("web.study_flow.timezone.now", return_value=now):
            response = self.client.get(reverse("web:study_timer"))
        self.assertEqual(response.json()["redirect"], reverse("web:study_post"))
        self.assertTrue(response.json()["expired"])
        session.refresh_from_db()
        self.assertEqual(session.stage, ParticipantSession.STAGE_POST)
        self.assertFalse(session.finished_early)

    def test_reopened_tutorial_returns_to_running_task(self):
        self.enter_task()
        response = self.client.get(reverse("web:study_tutorial"))
        self.assertContains(response, 'href="{}" class="begin"'.format(reverse("web:study_task")))
        self.assertNotContains(response, 'action="{}"'.format(reverse("web:study_begin")))

    def test_clock_resumes_after_absence_without_charging_time_away(self):
        self.enter_task()
        now = timezone.now()
        session = ParticipantSession.objects.get(user=self.user)
        session.task_elapsed_s = 10
        session.last_seen_at = now - dt.timedelta(seconds=120)
        session.save(update_fields=["task_elapsed_s", "last_seen_at"])

        with mock.patch("web.study_flow.timezone.now", return_value=now):
            study_flow.touch_task(session)
        session.refresh_from_db()
        self.assertEqual(session.task_elapsed_s, 10)
        self.assertEqual(session.last_seen_at, now)

        with mock.patch("web.study_flow.timezone.now", return_value=now + dt.timedelta(seconds=30)):
            study_flow.touch_task(session)
        session.refresh_from_db()
        self.assertEqual(session.task_elapsed_s, 40)
