# Prefer "docker compose" (plugin), fallback to "docker-compose" (standalone)
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# Load variables from .env
-include .env
export VERSION ?= dev
export ASSETS_DATE ?= dev
VERSION ?= dev
ASSETS_DATE ?= dev
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
	$(DOCKER_COMPOSE) up -d api web postgres # solr-api solr zoo solr_config

rebuild-run: rebuild-api rebuild-web # rebuild-solr-api
	$(DOCKER_COMPOSE) up -d api web postgres # solr-api

up: build
	$(DOCKER_COMPOSE) up -d api web postgres # solr-api solr zoo solr_config

down:
	$(DOCKER_COMPOSE) down --remove-orphans

logs-api:
	$(DOCKER_COMPOSE) logs -f api

logs-web:
	$(DOCKER_COMPOSE) logs -f web

logs-postgres:
	$(DOCKER_COMPOSE) logs -f postgres

# logs-solr-api:
# 	docker compose logs -f solr-api

.PHONY: build-builder build-assets rebuild-builder rebuild-assets \
        build-api build-web build-solr-api build-solr \
        rebuild-api rebuild-web rebuild-solr-api rebuild-solr \
        build rebuild-all rebuild-run up down \
        logs-api logs-web logs-postgres