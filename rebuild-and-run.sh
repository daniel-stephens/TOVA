#!/bin/bash
# Rebuild and run TOVA app

set -e
# Prefer "docker compose" (plugin), fallback to "docker-compose" (standalone)
if docker compose version &>/dev/null; then DC="docker compose"; else DC="docker-compose"; fi

echo "=== Cleaning up Docker ==="
docker system prune -a -f --volumes
docker builder prune -a -f

echo ""
echo "=== Rebuilding services without cache ==="
$DC build --no-cache api web

echo ""
echo "=== Starting services ==="
$DC up -d api web postgres

echo ""
echo "=== Services started ==="
echo "API: http://localhost:8000"
echo "Web UI: http://localhost:8080"
echo ""
echo "To view logs:"
echo "  $DC logs -f web"
echo "  $DC logs -f api"

