# Overnight work: IRB conformance and study tooling

Branch `study-irb-compliance`, off `userstudy`. Eleven commits, one per item or
group, so any single change can be reviewed or reverted on its own.

**Test suite: 234 passing, 0 failing.** Baseline before starting was 153 tests,
149 passing — the four failures were an environment artifact, described below.

## Environment

Tests run in the UI container with the source bind-mounted over the image,
because `Dockerfile.ui` does `COPY ui/ /app` and compose mounts only `./data`
and `./static/config` — so `docker compose exec` would test the image's baked-in
copy, not the working tree:

```
docker compose run --rm --no-deps \
  -v "$PWD/ui:/app" -v "$PWD/templates:/app/templates" -v "$PWD/src:/app/src" \
  -e DATABASE_URL= \
  web python manage.py test web
```

`--no-deps` skips the api and postgres services; clearing `DATABASE_URL` puts
the test database on SQLite inside the container. No local virtualenv was
created — the container is the environment, as you said.

**Not verified:** anything requiring a live backend. `_study_corpus_size()` and
the pre-flight page's document counts are tested against mocked responses (as
the existing suite already does for these paths); their behaviour against the
real API is unconfirmed. The morning checklist at the end lists what to click.

## What changed

### 1. Participants could skip the entire study (`a99c1e2`)

`StudyFlowMiddleware` returned early unless `study_group_id` was set. But
`is_study_participant()` — which every other part of the study code uses —
counts a directly-assigned `study_model_id` on its own, and **both** admin paths
that create participants (the "Create participants" form and the "Assign
dashboard" action) set `study_model_id` and never a group.

So the participants the research team actually creates were never stage-gated.
A `GET /study/post` skipped the exploration task outright. Every pre-existing
gating test built its participant *with* a group, which is why nothing caught it.

Now gated on `is_study_participant()`. I verified the three new tests fail
against the old check before keeping them.

### 2. Consent stage (`1c880d5`)

There was no consent step at all. Added as stage one, ahead of instructions:
`consented_at` + `consent_version` on the session, `STAGE_CONSENT` at the head
of `STAGE_ORDER`, nothing else reachable before consenting, idempotent
recording, and `CONSENT_VERSION` stamped per participant.

The consent text is reproduced **verbatim**; the design work is presentational
only — a summary band marked as orientation, an investigator block, print
styles. `test_renders_the_approved_text_verbatim` asserts all six approved
paragraphs word for word against the rendered page, so a future paraphrase fails
the build.

### 3. The questionnaire refused to let people skip (`e429988`)

The form rejected any blank field with a 400 and "Please answer this question",
while the approved consent says "You may skip a question". The software was
compelling answers the document said were optional.

Blanks are now recorded rather than refused; only an unrecognised *value* is an
error. Added "Prefer not to answer" (stored as `declined`, distinct from
`unsure` — "would rather not say" and "don't know" are different findings), and
`StudyResponse.tf_answered` so a low score from skipping is distinguishable from
a low score from error.

**Test whose expectations I changed:**
`test_missing_answers_are_rejected_and_input_preserved` asserted the old
behaviour. It is replaced by `test_blank_answers_are_accepted_because_the_consent_permits_skipping`
plus five others covering the new contract, including that garbage values are
still rejected. This is the only test in the suite whose meaning I reversed, and
it was reversed deliberately.

### 4. No way to get paid (`3e7fcef`)

`done.html` ended at "close this window", but Prolific only credits a submission
that returns through the completion URL. Added `PROLIFIC_COMPLETION_CODE` and
`PROLIFIC_RETURN_URL`, both optional; with neither set the page reads as before,
so the Upwork arm is unaffected.

### 5. Emails were in the analytic dataset (`e93f4fe`)

The consent says data is "linked only to a study-specific code". Both CSV
exports wrote `user.email` into every row, and the keyboard-vs-mouse comparison
rolled up by email.

Added `User.participant_code`, set automatically for generated logins
(`p001@study.local` → `p001`) by both the admin form and the CLI. Exports go
through `User.export_id`, which prefers the code and otherwise falls back to the
row's random UUID — so even a participant invited by real email exports
de-identified. A real address never becomes a code, because the local part of a
work address is usually a name.

### 6. The outcome data had no export (`e4a8c64`)

Reading the open answers meant clicking into each record one at a time, while
the telemetry had both a CSV and a rollup page. Added a streaming per-participant
CSV: study code, dashboard, condition, consent version, task timings, each
statement with a correctness flag, score, answered-count, each open answer with
its citation IDs. Flat, no JSON to unpack, no email anywhere.

### 7. Instructions contradicted the consent (`3ea0bc1`)

The instructions said "You'll still be able to look documents up while you
answer the questions"; the consent says the emails cannot be reopened. The code
was already the compliant half — the picker serves only IDs, subjects and dates.
Only the copy was wrong. Also added the plain statement that interactions are
logged. ("Not sure" was already the visible label for `unsure`; a test now pins
it.)

### 8. Moderated arm (`ce6a054`)

`ParticipantSession.moderated` plus a `ResearcherNote` model with a separate
author FK. Notes survive their author's account being deleted. `ResearcherNote`
is the one writable study model in the admin, since notes are typed live.

### 9. Instrument versioning and per-condition exposure (`95a95db`)

`tf_score` is computed at submission against whatever the key said then; a
mid-study revision would make old and new scores silently incomparable. The
instrument moved to `web/instruments.py` behind `INSTRUMENT_VERSION`, stamped on
each response. `task_minutes` became a nullable override on `StudyGroup` and
`User`, snapshotted at task start — the default stays 30, as the protocol
specifies.

### 10. Pre-flight and progress pages (`bc1c979`)

`/admin/tova/preflight` — every dashboard a participant could reach, with the
silent failures: nobody assigned, corpus unreadable, model missing from the
store. Degrades rather than erroring when the backend is down, since that is the
condition it exists to report.

`/admin/tova/progress` — the funnel per condition (assigned → consented →
started → in questions → complete) with mean score, mean answered and mean
duration.

### 11. Edge cases (`b255de8`)

Two real defects found while probing:

- `participant_code_for()` could hand out a code another account already owns
  (`p001@study.local` and `p001@example.com` both derive `p001`). The column is
  unique, so the second creation would raise `IntegrityError` and 500 the
  create-participants page **mid-cohort**. It now yields the code instead.
- Adding the consent stage changed the default for *new* sessions only, so a row
  already at "instructions" in a deployed database would walk past consent.
  Migration `0011` moves un-started sessions back to the gate and leaves runs
  already underway alone. Verified against a database pinned at `0010`: the
  un-started session moves, the task/post_study/done sessions do not.

Also covered: `?next=` cannot skip the flow, telemetry still posts at the
consent gate, `task_minutes=0` falls back rather than ending instantly,
double-finish does not re-advance, deleting a participant removes every trace of
their research data, unknown event actions are still dropped.

### 12. Active learning keyboard defects (`48cf586`)

Keyboard-vs-mouse is a measured outcome, so a mislabelled shortcut biases the
measurement — it turns a preference question into a discoverability question.
The hint advertised keys 1–3 while the handler accepts 1–9; it carried
`aria-hidden="true"`, so screen-reader users were never told shortcuts existed;
and each chip's `aria-label` hardcoded "of 3", so a fourth chip announced
"Suggestion 4 of 3".

### 0. Test portability (`03313dc`)

Four telemetry tests read the collector's source with a CWD-relative
`"../templates/..."` path that only resolves when run from `ui/`. Resolved
through the configured template dirs instead. No change to what they assert.

## Migrations

`0006` consent stage · `0007` tf_answered · `0008` participant_code ·
`0009` moderated sessions + ResearcherNote · `0010` condition overrides +
instrument_version · `0011` data migration for existing sessions.

All eleven apply cleanly from an empty database, and `makemigrations --check`
reports nothing missing.

## Deploy checklist

1. Apply migrations against Docker/Postgres — `DATABASE_URL` points the UI
   container at the Postgres service, and that volume is where the IRB-governed
   data physically sits. **Worth answering the backup question before recruiting.**
2. Set `PROLIFIC_COMPLETION_CODE` (and optionally `PROLIFIC_RETURN_URL`) for the
   Prolific arm. Leave unset for Upwork.
3. Set `TOVA_ALLOW_SIGNUP=false` for a study deployment, so the only way in is an
   account the research team created.
4. Confirm the consent wording against the approved version and bump
   `CONSENT_VERSION` if it differs.

## One thing I did not change

`manage.py check --deploy` reports five pre-existing warnings, none caused by
this work: `SECURE_HSTS_SECONDS` unset, `SECURE_SSL_REDIRECT` not True, a short
`SECRET_KEY`, and `SESSION_COOKIE_SECURE` / `CSRF_COOKIE_SECURE` not True.

All five are environment configuration rather than code — `SESSION_COOKIE_SECURE`
already follows `USE_HTTPS`, and `DJANGO_SECRET_KEY` is read from env. I left
them alone because forcing them on would break local development.

They are worth raising anyway: Prolific participants reach this over the public
internet, and the session carries consent records and questionnaire responses.
Before recruiting, the deployment wants `USE_HTTPS=true`, a real
`DJANGO_SECRET_KEY`, and TLS termination in front of it.

## To verify in the morning, with the backend up

- Walk the whole flow as a real participant: consent → instructions → task →
  questionnaire → done. Check the timer, and that the AL tab appears only for
  participants in that condition.
- Print the consent page and confirm the print view is legible.
- `/admin/tova/preflight` — confirm the assigned dashboards resolve to the
  ~600-email CorpusBio corpus and report "Ready".
- Download both CSVs and confirm the columns are what you want for analysis.

## Still open, and not mine to decide

The three PI questions in `irb-conformance.md`: Finish Early vs. the fixed
30-minute period; whether subject-line browsing counts as "from memory"; and
which consent version is approved.
