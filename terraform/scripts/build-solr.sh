#!/bin/bash
# Build Solr images for ECS deployment
# Usage: ./scripts/build-solr.sh [version]

set -e

# Get version from argument or .env file
if [ -n "$1" ]; then
    VERSION=$1
elif [ -f "../../.env" ]; then
    VERSION=$(grep "^VERSION=" ../../.env | cut -d'=' -f2)
    if [ -z "$VERSION" ]; then
        VERSION="dev"
    fi
else
    VERSION="dev"
    echo "Warning: No version specified and .env not found. Using 'dev' as version."
fi

echo "Building Solr images with version: $VERSION"

# Ensure we're in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Build builder image first if it doesn't exist
if ! docker images | grep -q "tova-builder:${VERSION}"; then
    echo "Building builder image..."
    docker build \
        --target builder \
        -t tova-builder:${VERSION} \
        -f docker/Dockerfile.base .
fi

# Build Solr API image
echo "Building Solr API image..."
docker build \
    --build-arg BUILDER_IMAGE=tova-builder:${VERSION} \
    -t tova-solr-api:${VERSION} \
    -f docker/Dockerfile.solr_api .

# Build Solr image
echo "Building Solr image..."
docker build \
    -t tova-solr:${VERSION} \
    -f docker/Dockerfile.solr_db .

echo ""
echo "Done! Solr images built:"
echo "  - tova-solr-api:${VERSION}"
echo "  - tova-solr:${VERSION}"
echo ""
echo "To tag as 'latest', run:"
echo "  docker tag tova-solr-api:${VERSION} tova-solr-api:latest"
echo "  docker tag tova-solr:${VERSION} tova-solr:latest"

