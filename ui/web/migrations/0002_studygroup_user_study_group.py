# Generated manually for study group support.

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("web", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="StudyGroup",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=100, unique=True)),
                ("slug", models.SlugField(max_length=100, unique=True)),
                ("description", models.TextField(blank=True)),
                (
                    "default_model_id",
                    models.CharField(
                        blank=True,
                        help_text="Trained model ID all users in this group land on after login.",
                        max_length=255,
                    ),
                ),
                ("enable_active_learning", models.BooleanField(default=True)),
            ],
            options={
                "db_table": "web_study_groups",
                "ordering": ["name"],
            },
        ),
        migrations.AddField(
            model_name="user",
            name="study_group",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="users",
                to="web.studygroup",
            ),
        ),
    ]
