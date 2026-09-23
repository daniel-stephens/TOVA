# Deploy the TOVA user study with Ollama

This guide runs the `unix` branch from a fresh checkout using Docker Compose,
PostgreSQL, local LDA training, and Ollama with `gemma3:4b`. It covers:

**Import emails → train LDA → generate topic names and summaries → create study
accounts → consent → instructions → tutorial → labeling → Finish labeling →
post-study questionnaire → completion and export.**

The macOS host-Ollama path was exercised with 1,151 email records and 10 topics.
The Linux container and public HTTPS examples below are deployment instructions;
they have not been exercised on a remote server in this work. No OpenAI API key
is required for this LDA/Ollama workflow. Other topic-model families have
separate dependencies.

## 1. Check out the branch and supply the data

Install Git, `make`, Python 3, Docker with Compose v2, and Ollama for the native
macOS option. Start Docker Desktop on macOS, or the Docker service on Linux.
Allow disk space for the Python build dependencies and Ollama model; a first
image build and model pull require internet access.

On a new machine:

```bash
git clone --branch unix https://github.com/daniel-stephens/TOVA.git
cd TOVA
mkdir -p data/drafts data/models
```

For an existing checkout, switch to `unix` after saving any unrelated work.
The deployment machine needs its own copy of the email collection:

```text
TOVA/data/email_corpus (1).jsonl
```

The dataset, trained models, database volume, `.env`, and generated credentials
are local artifacts; Git does not transfer them. On the original workstation,
the source is already at
`/Users/zongxiali/Downloads/My_Publication/auto_research/TOVA/data/email_corpus (1).jsonl`.
On another machine, transfer it separately into `data/`.

## 2. Create the environment

Copy the template only if `.env` does not already exist:

```bash
test -f .env || cp sample.env .env
chmod 600 .env
```

Edit `.env`. For a local browser run, use the following settings, replacing the
two secret placeholders with separately generated values:

```dotenv
VERSION=latest
ASSETS_DATE=latest
WEB_PORT=127.0.0.1:8080
API_PORT=127.0.0.1:11000
POSTGRES_PORT=127.0.0.1:5432
POSTGRES_USER=tova_user
POSTGRES_PASSWORD=REPLACE_WITH_RANDOM_VALUE
POSTGRES_DB=tova_db
DJANGO_SECRET_KEY=REPLACE_WITH_DIFFERENT_RANDOM_VALUE
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
CSRF_TRUSTED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
USE_HTTPS=false
TOVA_ALLOW_SIGNUP=false
TOVA_ADMIN_EMAILS=
STUDY_TASK_MINUTES=30
TOVA_ENABLE_AL3=false
OPENAI_API_KEY=
```

Generate each secret locally with `python3 -c 'import secrets;
print(secrets.token_hex(32))'`. Hex values avoid special-character parsing in
the Makefile and database URL. Retain the existing PostgreSQL credentials if
reusing a database volume. `prepare_study` creates its own researcher account,
so signup and environment-based administrator bootstrapping are unnecessary.

The port variables are substituted into Compose port mappings, so an explicit
`127.0.0.1:PORT` binds to loopback. If port 5432 is occupied, for example, use
`POSTGRES_PORT=127.0.0.1:5433`; the database's internal port remains 5432.

For Prolific, set the supplied `PROLIFIC_COMPLETION_CODE` and/or
`PROLIFIC_RETURN_URL` before running participants. Leave these blank for a local
walkthrough.

## 3. Build and start the application

Define this convenience function in each terminal used for the guide:

```bash
dc() {
  docker compose -f docker-compose.yaml -f docker-compose.dev.yml "$@"
}
```

Build images in dependency order and start the three study services:

```bash
make build
dc up -d --no-build --wait api web postgres
dc ps
curl --fail http://localhost:11000/health
```

Use `make build` on the first installation or when dependencies change. With
existing compatible images, start at `dc up`. The development override runs
the checked-out source, mounts templates and static assets, sets the shared
draft-store path, and avoids preloading embedding models for LDA. Keep this
checkout available while the containers run. Freeze it for participant runs.

The web entrypoint applies Django migrations and collects static files. Solr
and an embedding service are not required for this workflow. All remaining
Compose commands in this guide use `dc` so the same overrides stay in effect.

## 4. Start Ollama and install the model

Choose one of these options.

### Option A: native Ollama on macOS

For a Homebrew installation:

```bash
brew services start ollama
ollama pull gemma3:4b
ollama list
```

With the Ollama application, start that application instead of the Homebrew
service. With a standalone CLI installation, keep `ollama serve` running in a
separate terminal. Do not start a second server on the same port.

The Docker containers use `http://host.docker.internal:11434` to reach the host.
Native Ollama is the macOS option here because Docker Desktop does not provide
Ollama GPU passthrough on macOS. See the [Ollama FAQ](https://docs.ollama.com/faq).

### Option B: Ollama in Docker on Linux

After step 3 has created `tova_my_network`, start Ollama on that private network:

```bash
docker run -d --name tova-ollama --restart unless-stopped \
  --network tova_my_network --network-alias ollama \
  -v tova_ollama:/root/.ollama \
  ollama/ollama
docker exec tova-ollama ollama pull gemma3:4b
docker exec tova-ollama ollama list
```

This CPU example publishes no Ollama host port. For NVIDIA acceleration, install
the NVIDIA Container Toolkit and add `--gpus all` to `docker run`, following
[Ollama's Docker instructions](https://docs.ollama.com/docker). On subsequent
starts, use `docker start tova-ollama` instead of creating another container.

Set the application server URL to `http://ollama:11434` for this option.
Keep using `dc stop`/`dc start` for the application so its Docker network remains
available to the separately managed Ollama container.

## 5. Configure and verify the LLM connection

Edit these existing fields in `static/config/config.yaml`; retain all other
sections and the existing prompt paths:

```yaml
topic_modeling:
  general:
    llm_provider: ollama
    llm_model_type: gemma3:4b
    llm_server: http://host.docker.internal:11434
  traditional:
    do_labeller: true
    do_summarizer: true
llm:
  ollama:
    host: http://host.docker.internal:11434
```

For Linux option B, use `http://ollama:11434` in both URL fields. Restart to load
configuration changes:

```bash
dc restart api web
```

Verify model availability and actual text generation **from inside the API
container**, where training runs:

```bash
dc exec -T api python - <<'PY'
import requests
import yaml

with open('/app/static/config/config.yaml') as handle:
    cfg = yaml.safe_load(handle)['topic_modeling']['general']
base = cfg['llm_server'].rstrip('/')
model = cfg['llm_model_type']
tags = requests.get(base + '/api/tags', timeout=10)
tags.raise_for_status()
available = {item['model'] for item in tags.json()['models']}
assert model in available, f'{model} is missing; installed: {sorted(available)}'
reply = requests.post(base + '/api/generate', json={
    'model': model, 'prompt': 'Reply with only the word ready.',
    'stream': False, 'options': {'num_predict': 32},
}, timeout=180)
reply.raise_for_status()
assert reply.json().get('response', '').strip(), 'Ollama returned no text'
print('Ollama model and generation check passed:', model)
PY
```

If the native server cannot be reached, check its bind address and container
connectivity. Ollama defaults to loopback and configures listening addresses
through `OLLAMA_HOST`; see [Ollama server configuration](https://docs.ollama.com/faq#how-do-i-configure-ollama-server).
Use the private Docker option on Linux when a host service is unnecessary.

## 6. Import all emails, train, and create accounts

```bash
dc exec web python manage.py prepare_study \
  --source '/data/email_corpus (1).jsonl' \
  --name 'CorpusBio study' \
  --topics 10 \
  --iterations 100 \
  --task-minutes 30
```

The command:

1. Parses the supplied email file, including adjacent JSON objects. The
   supplied collection has 1,151 records; repeated source IDs are preserved as
   `original_id`, with separate internal IDs for every record.
2. Creates a fresh dataset and corpus, with readable email headers and body.
3. Trains a 10-topic LDA model with seed 42 and automatic NLP preprocessing off.
4. Generates topic names and summaries through the configured Ollama model.
5. Waits for completion and stops enrollment if the job reports warnings.
6. Creates a researcher account and one participant assigned to a new study
   group with active learning enabled.

The command prints a directory such as `/data/study-runs/corpusbio-study-.../`.
On the host this is `data/study-runs/corpusbio-study-.../`:

| File | Purpose |
| --- | --- |
| `report.json` | Run state, training result, dataset/corpus/model IDs, group and account IDs |
| `credentials.json` | Separate researcher and participant logins; file mode 0600 |
| `documents.json` | Normalized records that can also be uploaded through the web UI |

Read the credentials locally. These files stay under the Git-ignored `data/`
directory. The first LLM request may take longer while Ollama loads the model.
For slower machines, increase the command's `--timeout` (seconds; default 1800).

Every invocation creates a new run. To add participants to the existing run,
use the admin page in step 7 instead of training again. `--limit 50` is available
for a small technical smoke test. `--no-enrich` explicitly skips Ollama and
leaves numbered topics and placeholder summaries; omit both for a complete
collection with generated descriptions.

## 7. Verify as the researcher

Sign in at [the administrator login](http://localhost:8080/admin/) using the
researcher credentials. Adjust the host port if configured differently.

1. Open `/trained-models` and the model's dashboard. Confirm there are 10 named
   topics, actual summaries, and 1,151 documents for the complete supplied file.
2. Check `/admin/tova/preflight/` for model, corpus, and account assignment.
3. Inspect `/admin/web/studygroup/` and confirm the intended default model,
   active-learning condition, and task duration.
4. For additional accounts, open `/admin/web/user/create-participants/`, select
   that group, and keep the generated credentials for distribution.

**Current questionnaire content:** the running `corpusbio-v1` questionnaire
still contains questions from a different email narrative. The revised
[questionnaire draft](docs/user-study/post-study-questionnaire-v2-draft.md) and
[researcher key](docs/user-study/post-study-questionnaire-v2-key.md) match the
loaded Xylos/Bio-Vault emails, but are not installed in the form. The software
flow can be rehearsed now; align and version the final questionnaire before
using its answers as research outcomes. Neither deployment nor model training
automatically applies those drafts.

## 8. Walk through the participant study

Use a private/incognito window with a fresh **participant** account. Researcher
accounts bypass participant stage restrictions.

Start at **[http://localhost:8080/study/login](http://localhost:8080/study/login)**.

1. Sign in with the participant credentials.
2. Read the consent page and make your own consent decision.
3. Read the instructions, then continue to the tutorial.
4. Start the task from the tutorial and explore the emails.
5. Assign document labels with **Save Label and Continue**. Saved labels and
   the active-learning classifier belong to that participant. The underlying
   LDA model is not retrained by these label edits.
6. Click **Finish labeling** and confirm. Labeling ends and `/study/post` opens;
   labeling every document is not required. Timer expiry also ends the task.
7. Answer or skip the post-study questions and click **Submit**.
8. Confirm the completion page at `/study/done` and any configured recruitment
   platform completion link.

A later login resumes the saved stage. For another walkthrough, create a fresh
participant rather than resetting or deleting collected responses. Questionnaire
text is saved on submission, not as you type. Test accounts should be identified
by their participant codes and excluded from research analysis.

## 9. Regenerate descriptions without retraining

For an existing model whose names are still `Topic 0`, `Topic 1`, etc., use its
ID from the dashboard URL or `report.json`:

```bash
dc exec api python -m tova.core.topic_enrichment --model-id MODEL_ID
dc restart api web
```

Replace `MODEL_ID` with the actual ID. The command generates all names and
summaries before replacing the saved files, backs up the old descriptions and
metadata under `enrichment-backups/`, and records the LLM and timestamp. It
does not change model weights, document assignments, or participant labels.
Restart clears process caches; reload the dashboard afterward. Finish this
preparation before participants begin so their descriptions stay consistent.

## 10. Host it for participants on other computers

`localhost` points to the computer running each browser. For remote participants,
run the stack on a reachable server and publish the **web application** through
HTTPS. Keep the API, PostgreSQL, and Ollama on local/private interfaces.

For example, install Caddy as a host service, point `study.example.org` at the
server, and allow incoming ports 80 and 443. Keep the loopback port mappings in
step 2 and change these `.env` values:

```dotenv
DJANGO_ALLOWED_HOSTS=study.example.org
CSRF_TRUSTED_ORIGINS=https://study.example.org
USE_HTTPS=true
DJANGO_DEBUG=false
TOVA_ALLOW_SIGNUP=false
```

Use this Caddyfile, replacing the example hostname:

```caddyfile
study.example.org {
    reverse_proxy 127.0.0.1:8080
}
```

Caddy can manage the site's HTTPS certificate when DNS and ports are configured.
Follow its [reverse-proxy guide](https://caddyserver.com/docs/quick-starts/reverse-proxy)
to install/start the service and load this file. Then recreate the application
containers to apply the environment changes:

```bash
dc up -d --no-build --wait api web postgres
```

The participant URL becomes `https://study.example.org/study/login`. Rehearse a
complete run through this exact HTTPS URL, including login, labeling, submission,
and export. Keep the host awake and the Docker/Ollama services running throughout
collection. This guide does not configure a cloud account, domain, or proxy on
the original workstation.

## 11. Export results and preserve the run

Researcher pages:

| URL path | Records |
| --- | --- |
| `/admin/tova/progress/` | Study stages and completion progress |
| `/admin/web/participantsession/` | Assignment, timing, consent, and stage |
| `/admin/web/labelevent/labels.csv` | Label events |
| `/admin/web/studyresponse/responses.csv` | Questionnaire responses and instrument versions |
| `/admin/web/interactionevent/events.csv` | Interaction events |

Preserve both PostgreSQL and `data/`: accounts and responses live in the former;
corpora, trained models, and participant classifier/label artifacts live in the
latter. A database backup alone is insufficient. Keep configuration and private
credentials with the deployment record.

For a coordinated local backup **between participant sessions**, from the repo
root in a terminal with `dc` defined:

```bash
umask 077
tova_backup_dir="../tova-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$tova_backup_dir"
dc stop web api
dc exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > "$tova_backup_dir/postgres.dump"
tar -czf "$tova_backup_dir/study-files.tar.gz" data static/config .env
git rev-parse HEAD > "$tova_backup_dir/commit.txt"
dc start api web
```

Check that the dump and archive completed successfully. Store the backup
privately: it contains credentials and participant data. Copy it to protected
storage outside the deployment machine. Do not run `make reset-db` or
`docker compose down -v` against a study you intend to keep.

## 12. Checks and troubleshooting

Application checks, using an isolated SQLite test database:

```bash
dc exec -T -e DATABASE_URL=sqlite:// web python manage.py test web --noinput
dc exec -T web python manage.py check
dc logs --tail=100 api web postgres
```

The source bind mounts make these tests exercise this checkout. The automated
suite uses mocked or synthetic study operations; perform the participant
walkthrough as well. Do not submit test answers using a real participant's
account.

| Symptom | Action |
| --- | --- |
| Ollama connection refused | Confirm the chosen server is running; run the container-side generation check in step 5. Match the URL to native-host or Docker Ollama. |
| Model not found | Run `ollama list` on the host or `docker exec tova-ollama ollama list`; pull the exact configured tag. |
| Numbered topics or placeholder summaries | Check both enrichment flags and training warnings; run step 9 for an existing model. |
| API startup is slow | Use both Compose files. The override disables eager embedding loading for this LDA workflow. |
| Cannot find the email source | Quote `/data/email_corpus (1).jsonl` inside Docker, not the workstation's `/Users/...` path. |
| Training times out | Inspect the job ID in `report.json` and API logs before starting another run; the API job may still be running. |
| Wrong page after participant login | Use a normal participant assigned to the study group. An existing participant resumes its current stage; use a fresh account for a new rehearsal. |
| HTTPS form gives a CSRF error | Match `DJANGO_ALLOWED_HOSTS` and the full HTTPS origin, then recreate the web container with `dc up -d`. |
| Static images are missing or old | Use the dev override's static mount and `dc restart web` to recollect current assets. |
| Port already in use | Change the relevant host port in `.env`, keep internal ports unchanged, and run `dc up -d`. |

For the full researcher and participant operating instructions, see
[the end-to-end study guide](docs/user-study/end-to-end.md).
