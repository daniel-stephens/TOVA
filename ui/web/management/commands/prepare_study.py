"""Import a corpus, train local LDA, and provision a study without overwrites."""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils.text import slugify

from web.models import StudyGroup, User
from web.services import runtime as R
from web.services.study_setup import load_documents


class Command(BaseCommand):
    help = "Import JSON/JSONL, train LDA with topic labels and summaries, and create a fresh study."

    def add_arguments(self, parser):
        parser.add_argument("--source", required=True)
        parser.add_argument("--name", default="CorpusBio study")
        parser.add_argument("--topics", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=100)
        parser.add_argument("--limit", type=int)
        parser.add_argument("--task-minutes", type=int, default=30)
        parser.add_argument("--timeout", type=int, default=1800)
        parser.add_argument("--no-enrich", action="store_true",
                            help="Skip LLM labels and summaries for an offline technical test.")

    def handle(self, *args, **opts):
        for key in ("topics", "iterations", "task_minutes", "timeout"):
            if opts[key] < 1:
                raise CommandError(f"--{key.replace('_', '-')} must be positive.")
        try:
            documents, import_report = load_documents(Path(opts["source"]), opts["limit"])
        except (OSError, ValueError) as exc:
            raise CommandError(f"Cannot import source: {exc}") from exc
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        run_id = f"{slugify(opts['name'])[:40] or 'study'}-{stamp}"
        output = R.DRAFTS_SAVE.parent / "study-runs" / run_id
        output.mkdir(parents=True, mode=0o700)
        credentials_path = output / "credentials.json"
        report = {"run_id": run_id, "source": opts["source"], **import_report, "status": "preparing"}

        def save_report():
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

        def api(method, route, **kwargs):
            response = requests.request(method, f"{R.API}{route}", timeout=(5, 120), **kwargs)
            if not response.ok:
                raise CommandError(f"API {route}: HTTP {response.status_code}: {response.text[:500]}")
            return response.json()

        save_report()
        self.stdout.write(f"Run directory: {output}", ending="\n")
        self.stdout.write(f"Importing {len(documents)} records; repeated source IDs: {import_report['repeated_source_ids']}. All records retained.")
        (output / "documents.json").write_text(json.dumps(documents, ensure_ascii=False) + "\n")
        try:
            api("GET", "/health")
            researcher_email = f"researcher-{stamp}@study.local"
            researcher_password = secrets.token_urlsafe(18)
            researcher = User.objects.create_superuser(
                email=researcher_email, password=researcher_password, name=opts["name"] + " researcher",
                auth_source="local",
            )
            credentials = {"researcher": {"email": researcher_email, "password": researcher_password}}
            # Save immediately so partial failures do not leave inaccessible accounts.
            def save_credentials():
                fd = os.open(credentials_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as handle:
                    json.dump(credentials, handle, indent=2)
                    handle.write("\n")
            save_credentials()
            dataset_id = api("POST", "/data/datasets", json={
                "name": opts["name"], "owner_id": str(researcher.pk), "documents": documents,
                "description": "Imported by prepare_study",
            })["draft_id"]
            report["dataset_id"] = dataset_id
            save_report()
            dataset = api("GET", f"/data/datasets/{dataset_id}")
            corpus_id = api("POST", "/data/corpora", json={
                "name": opts["name"], "owner_id": str(researcher.pk), "datasets": [dataset],
            })["draft_id"]
            report["corpus_id"] = corpus_id
            save_report()
            corpus = api("GET", f"/data/corpora/{corpus_id}")
            training_params = {
                "preprocess_text": False, "do_labeller": not opts["no_enrich"],
                "do_summarizer": not opts["no_enrich"],
                "num_topics": opts["topics"], "num_iters": opts["iterations"], "seed": 42,
            }
            job = api("POST", "/train/json", json={
                "model": "tomotopyLDA", "model_name": opts["name"] + " LDA",
                "corpus_id": corpus_id, "data": corpus["documents"],
                "training_params": training_params,
            })
            report.update(job, training_params=training_params, status="training")
            save_report()
            self.stdout.write(f"Training {job['model_id']} on {len(documents)} documents…")
            deadline, last = time.monotonic() + opts["timeout"], None
            while time.monotonic() < deadline:
                status = api("GET", f"/status/jobs/{job['job_id']}")
                message = (status["status"], status.get("message"))
                if message != last:
                    self.stdout.write(f"{message[0]}: {message[1] or ''}")
                    self.stdout.flush()
                    last = message
                if status["status"] == "succeeded":
                    if not opts["no_enrich"] and (status.get("result") or {}).get("warnings"):
                        raise CommandError("Training completed with warnings; resolve these before enrolling participants: "
                                           + "; ".join(status["result"]["warnings"]))
                    break
                if status["status"] in {"failed", "cancelled"}:
                    raise CommandError(status.get("error") or status.get("message") or "Training failed")
                time.sleep(2)
            else:
                raise CommandError("Timed out waiting for training. The API job may still be running; see report.json for its ID.")
            model_id = job["model_id"]
            api("POST", f"/data/corpora/{corpus_id}/add_model", params={"model_id": model_id})
            api("GET", f"/data/models/{model_id}")
            password = secrets.token_urlsafe(18)
            with transaction.atomic():
                group = StudyGroup.objects.create(
                    name=(opts["name"][:55] + " " + stamp), slug=run_id,
                    default_model_id=model_id, enable_active_learning=True,
                    task_minutes=opts["task_minutes"],
                )
                participant = User.objects.create_user(
                    email=f"participant-{stamp}@study.local", password=password,
                    name="Study participant", participant_code=f"p-{stamp}",
                    study_group=group, auth_source="local",
                )
            credentials["participant"] = {"email": participant.email, "password": password}
            save_credentials()
            report.update(status="ready", group_id=group.pk, participant_id=str(participant.pk),
                          researcher_id=str(researcher.pk), task_minutes=opts["task_minutes"],
                          training_result=status.get("result"), credentials=str(credentials_path))
            save_report()
            self.stdout.write(self.style.SUCCESS(f"Ready. Credentials: {credentials_path}"))
            self.stdout.write("Researcher: /admin/tova/  Participant: /study/login")
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            save_report()
            raise CommandError(f"Setup stopped: {exc}. Saved progress: {output / 'report.json'}") from exc
