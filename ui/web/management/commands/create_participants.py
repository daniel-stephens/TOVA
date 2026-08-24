"""Create study participants and assign each one a dashboard.

A dashboard is a trained model. The study "grouping" is simply everyone who
shares one, so assigning dashboards is the same act as forming conditions.

Examples
--------
List what dashboards are available::

    python manage.py create_participants --list-dashboards

Spread 30 generated accounts evenly across every available dashboard::

    python manage.py create_participants --count 30 --dashboards auto --out creds.csv

Put 10 accounts on one specific dashboard, active learning off::

    python manage.py create_participants --count 10 \\
        --dashboard m_4096895c4c2b4a7b88ba42d7c6ba8213 --active-learning off

Invite named people, round-robin across two dashboards::

    python manage.py create_participants --emails ana@lab.gov,ben@lab.gov \\
        --dashboards m_abc,m_def
"""

from __future__ import annotations

import csv
import secrets
import sys

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from web.admin import available_dashboards
from web.models import ParticipantSession, User


class Command(BaseCommand):
    help = "Create study participants and assign each one a dashboard (trained model)."

    def add_arguments(self, parser):
        who = parser.add_argument_group("who to create")
        who.add_argument("--count", type=int, help="Generate this many accounts.")
        who.add_argument("--emails", help="Comma-separated list of real email addresses.")
        who.add_argument("--emails-file", help="File with one email address per line.")
        who.add_argument(
            "--prefix", default="p", help="Prefix for generated logins (default: p)."
        )
        who.add_argument(
            "--domain",
            default="study.local",
            help="Domain for generated logins (default: study.local).",
        )
        who.add_argument(
            "--start",
            type=int,
            default=1,
            help="First number for generated logins. Raise it for a second cohort "
            "so the numbering does not collide (default: 1).",
        )

        what = parser.add_argument_group("what to assign")
        what.add_argument("--dashboard", help="Put everyone on this one model ID.")
        what.add_argument(
            "--dashboards",
            help="Comma-separated model IDs to round-robin across, or 'auto' for "
            "every dashboard currently available.",
        )
        what.add_argument(
            "--active-learning",
            choices=["on", "off", "default"],
            default="default",
            help="Active learning for these participants. 'default' leaves it "
            "unset, which means on (default: default).",
        )

        how = parser.add_argument_group("how to run")
        how.add_argument(
            "--password",
            help="Use this password for everyone. Omit to generate a random one each.",
        )
        how.add_argument("--out", help="Write the credentials to this CSV file.")
        how.add_argument(
            "--skip-existing",
            action="store_true",
            help="Leave accounts that already exist untouched instead of reassigning them.",
        )
        how.add_argument(
            "--dry-run", action="store_true", help="Show what would happen, change nothing."
        )
        how.add_argument(
            "--list-dashboards",
            action="store_true",
            help="Print the available dashboards and exit.",
        )

    # ------------------------------------------------------------------ main

    def handle(self, *args, **opts):
        catalogue = available_dashboards()

        if opts["list_dashboards"]:
            self._print_dashboards(catalogue)
            return

        dashboards = self._resolve_dashboards(opts, catalogue)
        emails = self._resolve_emails(opts)
        active_learning = {"on": True, "off": False, "default": None}[opts["active_learning"]]

        assignments = [
            (email, dashboards[i % len(dashboards)]) for i, email in enumerate(emails)
        ]

        labels = dict(catalogue)
        self.stdout.write("")
        self.stdout.write(f"{'EMAIL':<34}{'DASHBOARD':<40}AL")
        self.stdout.write("-" * 84)
        for email, model_id in assignments:
            al = {True: "on", False: "off", None: "on (default)"}[active_learning]
            self.stdout.write(f"{email:<34}{model_id:<40}{al}")

        self.stdout.write("")
        for model_id in dashboards:
            n = sum(1 for _e, m in assignments if m == model_id)
            name = labels.get(model_id, model_id).split("  (")[0]
            self.stdout.write(f"  {name}: {n} participant(s)")

        if opts["dry_run"]:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Dry run -- nothing was created."))
            return

        rows = self._create(assignments, active_learning, opts)
        self._report(rows, opts)

    # ------------------------------------------------------------- resolvers

    def _print_dashboards(self, catalogue):
        if not catalogue:
            raise CommandError(
                "No dashboards available. Train a model first -- and note that "
                "models without a corpus_id are skipped, since their dashboard "
                "cannot render."
            )
        self.stdout.write("Available dashboards:")
        for model_id, label in catalogue:
            self.stdout.write(f"  {label}")

    def _resolve_dashboards(self, opts, catalogue) -> list[str]:
        if opts["dashboard"] and opts["dashboards"]:
            raise CommandError("Use either --dashboard or --dashboards, not both.")

        known = {model_id for model_id, _label in catalogue}

        if opts["dashboard"]:
            requested = [opts["dashboard"].strip()]
        elif opts["dashboards"]:
            if opts["dashboards"].strip().lower() == "auto":
                if not catalogue:
                    raise CommandError("--dashboards auto found no dashboards to use.")
                return [model_id for model_id, _label in catalogue]
            requested = [d.strip() for d in opts["dashboards"].split(",") if d.strip()]
        else:
            raise CommandError(
                "Specify --dashboard <model_id>, --dashboards <a,b,c>, or "
                "--dashboards auto. Run --list-dashboards to see the options."
            )

        unknown = [d for d in requested if d not in known]
        if unknown:
            raise CommandError(
                "Not a usable dashboard: "
                + ", ".join(unknown)
                + "\nEither the model does not exist or it has no corpus, which "
                "means its dashboard cannot render. Run --list-dashboards."
            )
        return requested

    def _resolve_emails(self, opts) -> list[str]:
        sources = [bool(opts["count"]), bool(opts["emails"]), bool(opts["emails_file"])]
        if sum(sources) != 1:
            raise CommandError("Specify exactly one of --count, --emails, or --emails-file.")

        if opts["emails"]:
            emails = [e.strip().lower() for e in opts["emails"].split(",") if e.strip()]
        elif opts["emails_file"]:
            try:
                with open(opts["emails_file"], encoding="utf-8") as fh:
                    emails = [ln.strip().lower() for ln in fh if ln.strip()]
            except OSError as exc:
                raise CommandError(f"Could not read {opts['emails_file']}: {exc}")
        else:
            if opts["count"] < 1:
                raise CommandError("--count must be at least 1.")
            start = opts["start"]
            emails = [
                f"{opts['prefix']}{n:03d}@{opts['domain']}"
                for n in range(start, start + opts["count"])
            ]

        if not emails:
            raise CommandError("No email addresses to create.")

        seen, unique = set(), []
        for email in emails:
            if email in seen:
                self.stderr.write(self.style.WARNING(f"Skipping duplicate: {email}"))
                continue
            seen.add(email)
            unique.append(email)
        return unique

    # -------------------------------------------------------------- creation

    def _create(self, assignments, active_learning, opts) -> list[dict]:
        rows = []
        with transaction.atomic():
            for email, model_id in assignments:
                user = User.objects.filter(email=email).first()

                if user is not None and opts["skip_existing"]:
                    rows.append(
                        {
                            "email": email,
                            "password": "(unchanged)",
                            "dashboard": user.study_model_id or "(unchanged)",
                            "status": "skipped",
                        }
                    )
                    continue

                password = opts["password"] or secrets.token_urlsafe(9)
                created = user is None
                if created:
                    user = User(email=email, name=email, auth_source="local")

                user.study_model_id = model_id
                user.enable_active_learning = active_learning
                user.is_active = True
                user.set_password(password)
                user.save()

                # A pending run would hold a stale dashboard snapshot. Runs that
                # have already started are left alone so their data stays within
                # one condition.
                ParticipantSession.objects.filter(
                    user=user, task_started_at__isnull=True
                ).delete()

                rows.append(
                    {
                        "email": email,
                        "password": password,
                        "dashboard": model_id,
                        "status": "created" if created else "updated",
                    }
                )
        return rows

    def _report(self, rows, opts):
        created = sum(1 for r in rows if r["status"] == "created")
        updated = sum(1 for r in rows if r["status"] == "updated")
        skipped = sum(1 for r in rows if r["status"] == "skipped")

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Created {created}, updated {updated}, skipped {skipped}."
            )
        )

        if opts["out"]:
            try:
                with open(opts["out"], "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(
                        fh, fieldnames=["email", "password", "dashboard", "status"]
                    )
                    writer.writeheader()
                    writer.writerows(rows)
            except OSError as exc:
                raise CommandError(f"Could not write {opts['out']}: {exc}")
            self.stdout.write(
                self.style.WARNING(
                    f"Credentials written to {opts['out']}. This is the only copy of "
                    "the passwords -- store it somewhere safe and delete it when the "
                    "study is over."
                )
            )
        elif any(r["status"] != "skipped" for r in rows):
            self.stdout.write("")
            self.stdout.write("Passwords (not stored anywhere else):")
            writer = csv.DictWriter(
                sys.stdout, fieldnames=["email", "password", "dashboard", "status"]
            )
            writer.writeheader()
            writer.writerows(rows)
