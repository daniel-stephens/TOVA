# TOVA: Topic Visualization & Analysis

## Build and deploy

Everything happens through the Makefile that encapsulates all the docker-compose commands.

### Quick Start

```bash
make up      # Build and start all services
make down    # Stop all services
make logs-api # View API logs
```

### Available Commands

**Building Images:**

- `make build` - Build all images (builder, assets, api, web, solr-api)
- `make build-api` - Build only the API service
- `make build-web` - Build only the web UI service  
- `make build-solr-api` - Build only the Solr API service

**Rebuilding (no cache):**

- `make rebuild-all` - Rebuild everything and start services
- `make rebuild-run` - Rebuild runtime services (api, web, solr-api) and start
- `make rebuild-api`, `make rebuild-web`, `make rebuild-solr-api` - Rebuild individual services

**Running Services:**

- `make up` - Build (if needed) and start all services
- `make down` - Stop and remove all containers

**Monitoring:**

- `make logs-api` - Follow API logs
- `make logs-web` - Follow web UI logs  
- `make logs-solr-api` - Follow Solr API logs

### Services

The application consists of:

- **API** (port 8000) - Main FastAPI application
- **Web** (port 8080) - Web UI interface
- **Solr API** (port 8001) - Solr search interface
- **Solr** (port 8983) - Apache Solr search engine
- **Zookeeper** (ports 2180, 2181) - Coordination service for Solr

## Functionalities

1. Users can upload one or more datasets. Each dataset contains raw text documents, and multiple datasets can be merged into a single corpus for model training.