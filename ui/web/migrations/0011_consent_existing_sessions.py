"""Send un-started sessions back to the consent gate.

Adding the consent stage changed the default for *new* sessions only, so a row
already sitting at "instructions" in a deployed database would walk straight
past consent and into the task.

Anyone who has not started the exploration task is moved back to the consent
stage: they have lost nothing, and they must consent like everybody else. A run
already underway is left exactly where it is -- interrupting someone mid-task
would cost them their session, and the timer is already ticking.

If a deployment ever does have participants stranded mid-flow without a consent
record, that is a question for the PI rather than something to paper over here.
"""

from django.db import migrations


def send_unstarted_back_to_consent(apps, schema_editor):
    ParticipantSession = apps.get_model("web", "ParticipantSession")
    ParticipantSession.objects.filter(
        consented_at__isnull=True,
        task_started_at__isnull=True,
    ).exclude(stage="consent").update(stage="consent")


def noop_reverse(apps, schema_editor):
    """Nothing to undo: the stage they came from is the one they go back to."""


class Migration(migrations.Migration):

    dependencies = [
        ("web", "0010_condition_overrides"),
    ]

    operations = [
        migrations.RunPython(send_unstarted_back_to_consent, noop_reverse),
    ]
