# Docker deployment

## Build prerequisites (first time or after dependency changes)

The UI and API images expect two local images built from `docker/Dockerfile.base`:

```bash
# From the repository root
docker build --target builder -t tova-builder:latest -f docker/Dockerfile.base .
docker build --target assets -t tova-assets:latest -f docker/Dockerfile.base .
```

Set `VERSION` and `ASSETS_DATE` in `.env` (see `sample.env`) so they match these tags (defaults `latest` work with the commands above).

## Start the stack

From the repo root you can use the **Makefile** (recommended):

```bash
cp sample.env .env   # once; edit secrets and ports
make up              # builds tova-builder / tova-assets, then compose up --build
make check-docker    # optional: curl API + web health
```

Or invoke Compose directly:

```bash
docker compose --env-file .env up --build -d
```

- **API**: host port `${API_PORT:-11000}` (defaults to **11000** if unset).
- **Web UI**: host port `${WEB_PORT:-8080}` → container `8080` (Gunicorn serves `wsgi:application`).

## Check the web UI (Django)

```bash
curl -sf http://localhost:${WEB_PORT:-8080}/__django_health/
curl -sf http://localhost:${WEB_PORT:-8080}/check-backend
```

## Check if API serves requests

```bash
curl -i http://localhost:11000/docs 
```

```bash
curl -i http://localhost:11000/health
```

## Check if the UI container sees the API service

```bash
docker compose exec web sh -lc 'getent hosts api; printenv API_BASE_URL; curl -sf http://api:11000/health || curl -i http://api:11000/docs'
```