# Load variables from .env
include .env
export $(shell sed 's/=.*//' .env)

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
	docker compose build api

build-web:
	docker compose build web

# build-solr-api:
# 	docker compose build solr-api

rebuild-api:
	docker compose build --no-cache api

rebuild-web:
	docker compose build --no-cache web

# rebuild-solr-api:
# 	docker compose build --no-cache solr-api

# ---------- combos ----------
build: build-builder build-assets build-api build-web

rebuild-all: rebuild-builder rebuild-assets rebuild-api rebuild-web # rebuild-solr-api
	docker compose up -d api web postgres # solr-api solr zoo solr_config

rebuild-run: rebuild-api rebuild-web # rebuild-solr-api
	docker compose up -d api web postgres # solr-api

up: build
	docker compose up -d api web postgres # solr-api solr zoo solr_config

down:
	docker compose down --remove-orphans

logs-api:
	docker compose logs -f api

logs-web:
	docker compose logs -f web

logs-postgres:
	docker compose logs -f postgres

# logs-solr-api:
# 	docker compose logs -f solr-api

.PHONY: build-builder build-assets rebuild-builder rebuild-assets \
        build-api build-web \
        rebuild-api rebuild-web \
        build rebuild-all rebuild-run up down \
        logs-api logs-web logs-postgres