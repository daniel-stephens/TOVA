# Prefer "docker compose" (plugin), fallback to "docker-compose" (standalone)
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# Docker quick start (from repo root):
#   cp sample.env .env   # edit POSTGRES_*, secrets, WEB_PORT, etc.
#   make up              # builds tova-builder / tova-assets, then compose --build + up
#   make superuser       # Django admin bootstrap (needs DJANGO_SUPERUSER_* in .env; stack must be up)
#   make check-docker    # optional smoke checks (needs curl)
#
# Compose loads `.env` automatically; VERSION / ASSETS_DATE must match your tova-builder / tova-assets tags.

# Load variables from .env
-include .env
export VERSION ?= dev
export ASSETS_DATE ?= dev
VERSION ?= dev
ASSETS_DATE ?= dev
# Defaults match docker-compose.yaml (${API_PORT:-11000}, ${WEB_PORT:-8080})
API_PORT ?= 11000
WEB_PORT ?= 8080
export $(shell [ -f .env ] && sed 's/=.*//' .env)

# ---------- base / artifact images (not in compose) ----------
build-builder:
	docker build \
		--target builder \
		-t tova-builder:$(VERSION) \
		-f docker/Dockerfile.base .

build-assets:
	docker build \
		--target assets \
		-t tova-assets:$(ASSETS_DATE) \
		-f docker/Dockerfile.base .

rebuild-builder:
	docker build --no-cache \
		--target builder \
		-t tova-builder:$(VERSION) \
		-f docker/Dockerfile.base .

rebuild-assets:
	docker build --no-cache \
		--target assets \
		-t tova-assets:$(ASSETS_DATE) \
		-f docker/Dockerfile.base .

# ---------- runtime images (compose services) ----------
build-api:
	$(DOCKER_COMPOSE) build api

build-web:
	$(DOCKER_COMPOSE) build web

build-solr-api:
	docker build \
		--build-arg BUILDER_IMAGE=tova-builder:$(VERSION) \
		-t tova-solr-api:$(VERSION) \
		-f docker/Dockerfile.solr_api .

build-solr:
	docker build \
		-t tova-solr:$(VERSION) \
		-f docker/Dockerfile.solr_db .

rebuild-solr-api:
	docker build --no-cache \
		--build-arg BUILDER_IMAGE=tova-builder:$(VERSION) \
		-t tova-solr-api:$(VERSION) \
		-f docker/Dockerfile.solr_api .

rebuild-solr:
	docker build --no-cache \
		-t tova-solr:$(VERSION) \
		-f docker/Dockerfile.solr_db .

rebuild-api:
	$(DOCKER_COMPOSE) build --no-cache api

rebuild-web:
	$(DOCKER_COMPOSE) build --no-cache web

# rebuild-solr-api:
# 	docker compose build --no-cache solr-api

# ---------- combos ----------
build: build-builder build-assets build-api build-web

rebuild-all: rebuild-builder rebuild-assets rebuild-api rebuild-web # rebuild-solr-api
	$(DOCKER_COMPOSE) up -d --build api web postgres # solr-api solr zoo solr_config

rebuild-run: rebuild-api rebuild-web # rebuild-solr-api
	$(DOCKER_COMPOSE) up -d --build api web postgres # solr-api

# Base images first, then compose builds api/web and starts stack (recommended entrypoint).
up: build-builder build-assets
	$(DOCKER_COMPOSE) up -d --build api web postgres # solr-api solr zoo solr_config

check-docker:
	@curl -sf --connect-timeout 2 "http://127.0.0.1:$(API_PORT)/health" && echo " API ok" || (echo " API check failed"; exit 1)
	@curl -sf --connect-timeout 2 "http://127.0.0.1:$(WEB_PORT)/__django_health/" && echo " web ok" || (echo " web check failed"; exit 1)

down:
	$(DOCKER_COMPOSE) down --remove-orphans

logs-api:
	$(DOCKER_COMPOSE) logs -f api

logs-web:
	$(DOCKER_COMPOSE) logs -f web

logs-postgres:
	$(DOCKER_COMPOSE) logs -f postgres

# Django admin (web.User uses email as USERNAME_FIELD — set these in .env, not DJANGO_SUPERUSER_USERNAME).
# `superuser`: Postgres DB inside Compose (requires `make up` and migrations). `superuser-local`: ui/ DB (DATABASE_URL or sqlite).
# Uses sync_superuser to upsert by email and keep one password/identity source.
superuser:
	@test -n "$(DJANGO_SUPERUSER_EMAIL)" && test -n "$(DJANGO_SUPERUSER_PASSWORD)" || (echo "Set DJANGO_SUPERUSER_EMAIL and DJANGO_SUPERUSER_PASSWORD in .env"; exit 1)
	$(DOCKER_COMPOSE) exec -T web python manage.py sync_superuser

superuser-local:
	@test -n "$(DJANGO_SUPERUSER_EMAIL)" && test -n "$(DJANGO_SUPERUSER_PASSWORD)" || (echo "Set DJANGO_SUPERUSER_EMAIL and DJANGO_SUPERUSER_PASSWORD in .env"; exit 1)
	cd ui && python3 manage.py sync_superuser

# logs-solr-api:
# 	docker compose logs -f solr-api

.PHONY: build-builder build-assets rebuild-builder rebuild-assets \
        build-api build-web build-solr-api build-solr \
        rebuild-api rebuild-web rebuild-solr-api rebuild-solr \
        build rebuild-all rebuild-run up down check-docker \
        logs-api logs-web logs-postgres superuser superuser-local