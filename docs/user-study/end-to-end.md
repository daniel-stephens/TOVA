# Run an end-to-end TOVA study

The researcher prepares the data, trains a model, and assigns it to a study
group. Participants then sign in, read the consent and instructions, complete
the tutorial, label documents, and choose **Finish labeling** to open the
post-study questionnaire. Submitting the questionnaire completes their session.

This guide uses the supplied CorpusBio email collection and the local
`tomotopyLDA` baseline. It does not require an LLM API key. CTM, TopicGPT, and
OpenTopicRAG have separate runtime requirements and are not validated by this
LDA workflow.

## Start the services

Run these commands from the repository root with Docker Desktop running. If
`.env` is missing, copy `sample.env` to `.env` and configure the ports and
database credentials. Keep an existing `.env` and database volume together.

If the runtime images have not been built yet, build them once:

```bash
make build
```

Start PostgreSQL, the API, and the web application using the current source:

```bash
docker compose -f docker-compose.yaml -f docker-compose.dev.yml up -d --no-build api web postgres
docker compose -f docker-compose.yaml -f docker-compose.dev.yml ps
```

The development override mounts the source code and disables eager embedding
model loading. The first image build can download dependencies and model assets;
the LDA setup below disables automatic NLP preprocessing and uses local Ollama
to generate topic names and summaries.

Start Ollama on the host before training. On this Mac, it is installed with
Homebrew and already has `gemma3:4b`:

```bash
brew services start ollama
ollama list
```

For another installation, start the Ollama application or run `ollama serve`.
If `gemma3:4b` is missing, install it with `ollama pull gemma3:4b`.
`static/config/config.yaml` uses provider `ollama`, model `gemma3:4b`, and
server `http://host.docker.internal:11434` so the Docker API can reach the host.
When running the API directly on the host, use `http://localhost:11434` instead.

The default web address is [http://localhost:8080](http://localhost:8080); use
your `.env` `WEB_PORT` if different. The API health endpoint is
[http://localhost:11000/health](http://localhost:11000/health), using `API_PORT`
if different. Wait for the API to become healthy and the web service to start.
The web entrypoint applies database migrations.

## Prepare the supplied email corpus and train a model

The supplied source is `data/email_corpus (1).jsonl` on the host. Docker mounts
the repository's `data` directory at `/data`, so quote its container path:

```bash
docker compose -f docker-compose.yaml -f docker-compose.dev.yml exec web \
  python manage.py prepare_study \
  --source '/data/email_corpus (1).jsonl' \
  --name 'CorpusBio study' \
  --topics 10 \
  --iterations 100
```

The command imports a fresh dataset and corpus, trains a real LDA model, and
creates a new study group with active learning enabled plus separate researcher
and participant accounts. Each invocation creates a new run so the original
source and existing study assignments remain available. Ollama generates topic
names and summaries from each topic's keywords and representative emails before
participants are enrolled. The command stops enrollment if training reports a
warning, including a failure to generate labels or summaries. Participants
supply their own human labels during the task.

The email source contains 1,151 records with `mid`, `sender`, `recipient`,
`date`, `subject`, `body`, and `folder` fields. Some lines contain adjacent
JSON objects, so it is not strictly one JSON object per line. The command
accepts this format, retains every record, gives duplicate IDs unique document
IDs, and preserves the source ID as `original_id`. It builds model text from
the subject and body and readable email text from the headers and body.

The command prints the run directory. It contains:

| File | Purpose |
| --- | --- |
| `credentials.json` | Researcher and participant sign-in credentials |
| `report.json` | Prepared run details and dataset, corpus, and model identifiers |
| `documents.json` | Normalized document list suitable for web upload, with `id` and `text` fields |

The directory `/data/study-runs/<run-directory>/` inside Docker corresponds to
`data/study-runs/<run-directory>/` in the repository. Open the credentials file
locally and use its separate accounts; keep that file out of shared reports.

For a smaller technical test, add `--limit 50`. Omit `--limit` when preparing the
full collection. The command also accepts a JSON list of document objects or
a JSON object with a `documents` list, ordinary JSONL, or concatenated JSON
objects. A small subset is useful for testing the
workflow but can omit the evidence needed by the CorpusBio questionnaire.
For an explicitly offline technical test, add `--no-enrich`; that skips Ollama
and leaves numbered topic names and placeholder summaries.

## Generate names and summaries for an existing model

Use the following command to enrich a model without repeating LDA training.
Replace `MODEL_ID` with the identifier from the dashboard URL or run report:

```bash
docker compose -f docker-compose.yaml -f docker-compose.dev.yml exec api \
  python -m tova.core.topic_enrichment --model-id MODEL_ID
docker compose -f docker-compose.yaml -f docker-compose.dev.yml restart api web
```

The command generates all outputs before replacing the saved topic names and
summaries. It backs up the previous files and metadata under the model's
`enrichment-backups` directory and records the LLM and generation time in model
metadata. It preserves model weights, document assignments, and participant
labels. Restarting the services clears cached names and summaries; reload the
dashboard afterward. Complete enrichment before collecting participant data so
participants see the same topic descriptions.

## Researcher setup through the web interface

The command above performs these setup steps automatically. To use the web
interface for another model or collection, sign in with a researcher account
and follow this sequence. If you have not created an administrator, run:

```bash
docker compose -f docker-compose.yaml -f docker-compose.dev.yml exec web \
  python manage.py createsuperuser
```

1. Open **Upload data** at `/load-data-page/`. Choose **Single File**, select a
   normalized JSON document list, and map **ID Column** to `id` and **Text
   Column Name** to `text`. For this email collection, use the `documents.json`
   written by `prepare_study`; the original file's adjacent JSON objects do not
   meet the browser uploader's JSONL requirements. Give it a **Corpus Name**. An
   optional existing label column is separate from participants' study labels.
   Click **Save**. This saves both a dataset and a corpus.
2. Verify the new corpus appears under **Manage Corpora** at
   `/load-corpus-page/`. Its details should show the expected document count.
3. Open `/model`, select that corpus, choose `tomotopyLDA`, and click
   **Configure & Train**. Give the model a unique name and use 10 topics and
   `num_iters` of 100 for the baseline. Turn off **automatic preprocessing**.
   Enable **Generate topic labels with LLM** and **Generate topic summaries
   with LLM**, using the local Ollama settings above.
   Start training and wait for the completion result on `/training/`.
4. Open `/trained-models` and inspect the completed model's dashboard. Confirm
   that it contains topics and readable documents before assigning it.
5. Open `/admin/web/studygroup/add/`. Enter the group's name and slug, select
   the trained model as its default dashboard, and enable active learning.
   Save the group.
6. Open `/admin/web/user/create-participants/`. Enter the number of participants
   to generate, select the intended group, and create the accounts. Record the
   generated logins and passwords from the results. Participants must be normal
   accounts: staff and administrator accounts bypass the participant study flow.
7. Open `/admin/tova/preflight/` and check that the assigned dashboard, corpus,
   document count, participant assignment, and active-learning setting are ready.
   Use `/admin/tova/progress/` to follow collection progress.

Finish assigning the model and condition before a participant begins, and keep
those assignments unchanged during collection. Use a fresh participant account
for another complete trial.

## Participant instructions

Use a separate browser profile or private window so the researcher and
participant sessions do not replace one another. Give the participant the
`/study/login` address and their generated login details.

1. **Sign in:** open `/study/login` with the participant account.
2. **Consent:** read the consent page and select **I consent — continue** to
   proceed. Declining leaves the task unstarted.
3. **Instructions:** read what the task involves and the general kinds of
   questions that follow, then continue to the tutorial.
4. **Tutorial:** review the walkthrough and start the task using its start
   button. The task timer begins when the task starts.
5. **Labeling:** read a recommended document, choose or enter its label, and
   select **Save Label and Continue**. Continue with further documents and
   revise labels when needed. Each participant's saved labels and active
   learning classifier are separate, even when participants share a model.
6. **Finish labeling:** when ready, choose **Finish labeling** and confirm the
   transition. The application saves the task completion and opens
   `/study/post`. You do not have to label every document. Reaching the task
   time limit also moves the session to the questionnaire.
7. **Post-study questions:** complete the true/false/not-sure questions and
   written responses. Questions can be skipped. Click **Submit** to save the
   responses and move to `/study/done`.
8. **Completion:** confirm that the thank-you page appears. If a recruitment
   platform completion link is configured, follow the link displayed there.

The study proceeds in order. Once labeling has ended, the participant cannot
return to the exploration tool from the questionnaire. A later sign-in resumes
the saved study stage. Questionnaire text that has not been submitted is not a
saved response.

Human labeling updates the participant's separate active-learning classifier;
it does not retrain the underlying LDA topic model. Recommendations start
without a classifier and become classifier-driven after at least two distinct
label classes are available.

## Check the collected results

Sign in as the researcher and review these administrator pages:

| Page | What to check |
| --- | --- |
| `/admin/web/participantsession/` | Participant stage, task timing, completion, and model assignment |
| `/admin/web/labelevent/` | Saved labels, revisions, and removals |
| `/admin/web/studyresponse/` | Submitted questionnaire responses and instrument version |
| `/admin/web/interactionevent/` | Recorded interaction events |

The corresponding CSV downloads are
`/admin/web/labelevent/labels.csv`,
`/admin/web/studyresponse/responses.csv`, and
`/admin/web/interactionevent/events.csv`. Use the participant identifier to
relate the records.

PostgreSQL stores accounts, study sessions, questionnaire responses, and study
events. The `data/drafts` directory holds the imported data, trained models,
and participant-scoped active-learning labels. Keep both when preserving a run.

## If a step fails

- **Services do not start:** inspect
  `docker compose -f docker-compose.yaml -f docker-compose.dev.yml logs --tail=100 api web postgres`.
  If a host port is occupied, set `WEB_PORT`, `API_PORT`, or `POSTGRES_PORT` in
  `.env` and restart the services with the same Compose files.
- **Source file cannot be found:** `--source` runs inside the web container.
  Use `/data/...`, not the host's `/Users/...` path, and quote filenames with
  spaces or parentheses.
- **The participant sees the researcher interface:** use the generated
  participant credentials in a separate browser session and enter at
  `/study/login`. Confirm their group has an assigned model.
- **No documents or topics appear:** check the completed training job and
  `/admin/tova/preflight/`. A model must reference a readable corpus.
- **Topics are still numbered or summaries are placeholders:** confirm Ollama
  is running and `ollama list` includes the configured model. Check that both
  LLM generation options are enabled. For an already trained model, use the
  enrichment command above and reload the dashboard after restarting services.
- **Finishing reports a connection error:** wait for the connection to recover
  and retry **Finish labeling**. Previously saved labels are preserved.
- **A repeat test opens the questionnaire or completion page:** the account
  already reached that stage. Prepare a new run or generate a fresh participant
  in the same group for an independent trial.
