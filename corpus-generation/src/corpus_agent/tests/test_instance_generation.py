from unittest.mock import AsyncMock

import pytest

from corpus_state.db import create_database, export_instance_emails, get_instance
from corpus_state.types import Character, Email

from corpus_agent.config import PlotPoint
from corpus_agent.phases.instance_generation import run_instance
from corpus_agent.phases.wave_planner import EmailPlan, WavePlan


def _make_characters(conn):
    from corpus_state.db import upsert_character

    chars = [
        Character(
            id="alice",
            corpus_id="test",
            wire_character="A",
            synthetic_character="Alice Chen",
            email_address="a@b.com",
            role="Lead",
            organization="X",
            personality="driven",
            voice_anchor="formal",
        ),
        Character(
            id="bob",
            corpus_id="test",
            wire_character="B",
            synthetic_character="Bob Drake",
            email_address="b@b.com",
            role="CFO",
            organization="X",
            personality="calculating",
            voice_anchor="terse",
        ),
    ]
    for c in chars:
        upsert_character(conn, c)
    return chars


class TestRunInstance:
    @pytest.mark.asyncio
    async def test_generates_emails_across_waves(self):
        conn = create_database(":memory:")
        chars = _make_characters(conn)

        call_count = 0

        async def mock_plan_wave(**kwargs):
            nonlocal call_count
            call_count += 1
            return WavePlan(
                emails=[
                    EmailPlan(
                        sender_id="alice",
                        recipient_ids=["bob"],
                        is_reply=False,
                        parent_mid=None,
                        plot_point_to_reveal=f"p{call_count}"
                        if call_count <= 2
                        else None,
                        is_noise=call_count > 2,
                        intent=f"wave {call_count} email",
                    ),
                ],
                wave_rationale=f"wave {call_count}",
            )

        email_counter = 0

        async def mock_generate_email(**kwargs):
            nonlocal email_counter
            email_counter += 1
            return Email(
                sender="a@b.com",
                recipient="b@b.com",
                date=f"2026-05-15T{email_counter:02d}:00:00Z",
                message_id=f"<msg-{email_counter}@test.com>",
                subject=f"Email {email_counter}",
                body=f"Body {email_counter}",
                folder="\\Sent",
            )

        await run_instance(
            conn=conn,
            instance_id=1,
            corpus_id="test",
            characters=chars,
            plot_outline=[
                PlotPoint(id="p1", description="first"),
                PlotPoint(id="p2", description="second"),
            ],
            waves_per_instance=3,
            emails_per_wave=1,
            emails_per_instance=10,
            blur_level="medium",
            plan_wave_fn=mock_plan_wave,
            generate_email_fn=mock_generate_email,
            wave_pause_seconds=0,
        )

        inst = get_instance(conn, 1)
        assert inst.status == "completed"
        assert inst.num_emails == 3

        emails = export_instance_emails(conn, 1)
        assert len(emails) == 3

    @pytest.mark.asyncio
    async def test_invalid_parent_mid_is_dropped(self):
        conn = create_database(":memory:")
        chars = _make_characters(conn)

        async def mock_plan_wave(**kwargs):
            return WavePlan(
                emails=[
                    EmailPlan(
                        sender_id="alice",
                        recipient_ids=["bob"],
                        is_reply=True,
                        parent_mid=9999,
                        plot_point_to_reveal=None,
                        is_noise=False,
                        intent="reply with bad parent",
                    )
                ],
                wave_rationale="bad parent",
            )

        async def mock_generate_email(**kwargs):
            return Email(
                sender="a@b.com",
                recipient="b@b.com",
                date="2026-05-15T01:00:00Z",
                message_id="<msg-bad-parent@test.com>",
                subject="Email 1",
                body="Body 1",
                folder="\\Sent",
                parent_mid=9999,
            )

        await run_instance(
            conn=conn,
            instance_id=1,
            corpus_id="test",
            characters=chars,
            plot_outline=[PlotPoint(id="p1", description="first")],
            waves_per_instance=1,
            emails_per_wave=1,
            emails_per_instance=5,
            blur_level="medium",
            plan_wave_fn=mock_plan_wave,
            generate_email_fn=mock_generate_email,
            wave_pause_seconds=0,
        )

        emails = export_instance_emails(conn, 1)
        assert len(emails) == 1
        assert emails[0].parent_mid is None
