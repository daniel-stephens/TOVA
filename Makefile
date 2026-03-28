# Load variables from .env
include .env
export $(shell sed 's/=.*//' .env)

# colors
BOLD   := \033[1m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
CYAN   := \033[0;36m
RESET  := \033[0m

# base / artifact images (not in compose)
build-builder:
	@printf "$(CYAN)$(BOLD)» Building base builder image...$(RESET)\n"
	docker build \
		--target builder \
		-t tova-builder:$(VERSION) \
		-f docker/Dockerfile.base .
	@printf "$(GREEN)✔ Builder image ready$(RESET)\n"

build-assets:
	@printf "$(CYAN)$(BOLD)» Building assets image...$(RESET)\n"
	docker build \
		--target assets \
		-t tova-assets:$(ASSETS_DATE) \
		-f docker/Dockerfile.base .
	@printf "$(GREEN)✔ Assets image ready$(RESET)\n"

rebuild-builder:
	@printf "$(CYAN)$(BOLD)» Rebuilding base builder image (no-cache)...$(RESET)\n"
	docker build --no-cache \
		--target builder \
		-t tova-builder:$(VERSION) \
		-f docker/Dockerfile.base .
	@printf "$(GREEN)✔ Builder image rebuilt$(RESET)\n"

rebuild-assets:
	@printf "$(CYAN)$(BOLD)» Rebuilding assets image (no-cache)...$(RESET)\n"
	docker build --no-cache \
		--target assets \
		-t tova-assets:$(ASSETS_DATE) \
		-f docker/Dockerfile.base .
	@printf "$(GREEN)✔ Assets image rebuilt$(RESET)\n"

# runtime images (compose services)
build-api:
	@printf "$(CYAN)$(BOLD)» Building API image...$(RESET)\n"
	docker compose build api
	@printf "$(GREEN)✔ API image ready$(RESET)\n"

build-web:
	@printf "$(CYAN)$(BOLD)» Building web image...$(RESET)\n"
	docker compose build web
	@printf "$(GREEN)✔ Web image ready$(RESET)\n"

# build-solr-api:
# 	docker compose build solr-api

rebuild-api:
	@printf "$(CYAN)$(BOLD)» Rebuilding API image (no-cache)...$(RESET)\n"
	docker compose build --no-cache api
	@printf "$(GREEN)✔ API image rebuilt$(RESET)\n"

rebuild-web:
	@printf "$(CYAN)$(BOLD)» Rebuilding web image (no-cache)...$(RESET)\n"
	docker compose build --no-cache web
	@printf "$(GREEN)✔ Web image rebuilt$(RESET)\n"

# rebuild-solr-api:
# 	docker compose build --no-cache solr-api

# pre-flight checks
check-dirs:
	@mkdir -p data/drafts data/models
	@printf "$(GREEN)✔ Data directories ready$(RESET)\n"

check-ports:
	@printf "$(CYAN)$(BOLD)» Checking ports...$(RESET)\n"
	@failed=0; \
	for port in $(API_PORT) $(WEB_PORT); do \
		if ss -tlnp 2>/dev/null | grep -q ":$$port "; then \
			printf "$(RED)✖ Port $$port is already in use. Stop the process or change the port in .env$(RESET)\n"; \
			failed=1; \
		else \
			printf "$(GREEN)✔ Port $$port is free$(RESET)\n"; \
		fi; \
	done; \
	if [ $$failed -eq 1 ]; then exit 1; fi

# combos
build: build-builder build-assets build-api build-web
	@printf "$(GREEN)$(BOLD)✔ All images built$(RESET)\n"

rebuild-all: check-dirs check-ports rebuild-builder rebuild-assets rebuild-api rebuild-web # rebuild-solr-api
	@printf "$(CYAN)$(BOLD)» Starting services...$(RESET)\n"
	docker compose up -d api web postgres # solr-api solr zoo solr_config
	@printf "$(GREEN)$(BOLD)✔ Stack is up — web: http://$(HOST):$(WEB_PORT)  api: http://$(HOST):$(API_PORT)$(RESET)\n"

rebuild-run: check-dirs check-ports rebuild-api rebuild-web # rebuild-solr-api
	@printf "$(CYAN)$(BOLD)» Starting services...$(RESET)\n"
	docker compose up -d api web postgres # solr-api
	@printf "$(GREEN)$(BOLD)✔ Stack is up — web: http://$(HOST):$(WEB_PORT)  api: http://$(HOST):$(API_PORT)$(RESET)\n"

up: check-dirs check-ports build
	@printf "$(CYAN)$(BOLD)» Starting services...$(RESET)\n"
	docker compose up -d api web postgres # solr-api solr zoo solr_config
	@printf "$(GREEN)$(BOLD)✔ Stack is up — web: http://$(HOST):$(WEB_PORT)  api: http://$(HOST):$(API_PORT)$(RESET)\n"

# Smart entry point: detects uncommitted changes in build, relevant paths and
# offers a full rebuild before starting.  Falls back to 'make up' if clean.
smart-up:
	@CHANGED=$$(git diff --name-only HEAD 2>/dev/null | grep -E '^(docker/|ui/|src/|pyproject\.toml|requirements.*\.txt)'); \
	if [ -n "$$CHANGED" ]; then \
		printf "$(YELLOW)$(BOLD)⚠  Uncommitted changes detected in:$(RESET)\n"; \
		echo "$$CHANGED" | while read f; do printf "    $(YELLOW)$$f$(RESET)\n"; done; \
		printf "$(YELLOW)$(BOLD)Run a full rebuild? [y/N] $(RESET)"; \
		read ans; \
		case "$$ans" in \
			[yY]*) $(MAKE) rebuild-all ;; \
			*) $(MAKE) up ;; \
		esac; \
	else \
		printf "$(GREEN)✔ No relevant changes detected$(RESET)\n"; \
		$(MAKE) up; \
	fi

down:
	@printf "$(CYAN)$(BOLD)» Stopping stack...$(RESET)\n"
	docker compose down --remove-orphans
	@printf "$(GREEN)✔ Stack stopped$(RESET)\n"

reset-db:
	@printf "$(RED)$(BOLD)⚠  WARNING: this will delete ALL Postgres data.$(RESET)\n"
	@printf "$(RED)Ctrl-C to abort, Enter to continue...$(RESET)"; read _
	docker compose down --remove-orphans
	docker volume rm $$(docker volume ls -q | grep postgres_data) 2>/dev/null || true
	@printf "$(GREEN)✔ Postgres volume removed. Run 'make up' to reinitialize.$(RESET)\n"

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
        build rebuild-all rebuild-run up smart-up down \
        logs-api logs-web logs-postgres \
        check-ports check-dirs reset-db
