#!/bin/bash
# Docker cleanup and rebuild script
# Prefer "docker compose" (plugin), fallback to "docker-compose" (standalone)
if docker compose version &>/dev/null; then DC="docker compose"; else DC="docker-compose"; fi

echo "=== Removing unused Docker images ==="
docker image prune -a -f

echo ""
echo "=== Removing all unused containers, networks, images, and build cache ==="
docker system prune -a -f --volumes

echo ""
echo "=== Removing build cache ==="
docker builder prune -a -f

echo ""
echo "=== Rebuilding all images without cache ==="
# Rebuild all services without cache (adjust service names as needed)
$DC build --no-cache

echo ""
echo "=== Docker cleanup complete ==="
echo "Remaining images:"
docker images



