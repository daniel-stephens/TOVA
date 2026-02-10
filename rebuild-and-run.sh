#!/bin/bash
# Rebuild and run TOVA app

set -e

echo "=== Cleaning up Docker ==="
docker system prune -a -f --volumes
docker builder prune -a -f

echo ""
echo "=== Rebuilding services without cache ==="
docker compose build --no-cache api web

echo ""
echo "=== Starting services ==="
docker compose up -d api web postgres

echo ""
echo "=== Services started ==="
echo "API: http://localhost:8000"
echo "Web UI: http://localhost:8080"
echo ""
echo "To view logs:"
echo "  docker compose logs -f web"
echo "  docker compose logs -f api"

