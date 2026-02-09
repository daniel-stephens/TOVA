#!/bin/bash
# Build and push Docker images to ECR
# Usage: ./scripts/build-and-push.sh [region] [account-id]

set -e

REGION=${1:-us-east-1}
ACCOUNT_ID=${2:-""}
PROJECT_NAME="tova"

if [ -z "$ACCOUNT_ID" ]; then
    echo "Getting AWS account ID..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi

echo "Using AWS Account: $ACCOUNT_ID"
echo "Using Region: $REGION"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Get ECR repository URLs (you may need to run terraform output first)
API_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${PROJECT_NAME}-api"
WEB_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${PROJECT_NAME}-web"
SOLR_API_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${PROJECT_NAME}-solr-api"
SOLR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${PROJECT_NAME}-solr"

echo "Building base images..."
# Build base images if they don't exist
if ! docker images | grep -q "tova-builder"; then
    make build-builder
fi

if ! docker images | grep -q "tova-assets"; then
    make build-assets
fi

echo "Building application images..."
make build-api
make build-web

# Build Solr images (requires VERSION from .env)
if [ -f "../.env" ]; then
    echo "Building Solr images..."
    cd ..
    make build-solr-api
    make build-solr
    cd terraform/scripts || cd terraform
else
    echo "Warning: .env file not found. Solr images will not be built."
    echo "Please create .env file with VERSION and ASSETS_DATE, or build Solr images manually."
fi

echo "Tagging images..."
# Get version from .env or use 'latest'
VERSION_TAG="latest"
if [ -f "../.env" ]; then
    VERSION=$(grep "^VERSION=" ../.env | cut -d'=' -f2)
    if [ -n "$VERSION" ]; then
        VERSION_TAG="$VERSION"
    fi
fi

docker tag tova-api:${VERSION_TAG} $API_REPO:latest 2>/dev/null || docker tag tova-api:latest $API_REPO:latest
docker tag tova-web:${VERSION_TAG} $WEB_REPO:latest 2>/dev/null || docker tag tova-web:latest $WEB_REPO:latest

# Tag Solr images
if docker images | grep -qE "tova-solr-api.*${VERSION_TAG}"; then
    docker tag tova-solr-api:${VERSION_TAG} $SOLR_API_REPO:latest
elif docker images | grep -q "tova-solr-api"; then
    docker tag $(docker images tova-solr-api --format "{{.Repository}}:{{.Tag}}" | head -1) $SOLR_API_REPO:latest
fi

if docker images | grep -qE "tova-solr.*${VERSION_TAG}"; then
    docker tag tova-solr:${VERSION_TAG} $SOLR_REPO:latest
elif docker images | grep -q "tova-solr"; then
    docker tag $(docker images tova-solr --format "{{.Repository}}:{{.Tag}}" | head -1) $SOLR_REPO:latest
fi

echo "Pushing images to ECR..."
docker push $API_REPO:latest
docker push $WEB_REPO:latest

# Push Solr images if they exist
if docker images | grep -q "$SOLR_API_REPO:latest"; then
    echo "Pushing Solr API image..."
    docker push $SOLR_API_REPO:latest
fi

if docker images | grep -q "$SOLR_REPO:latest"; then
    echo "Pushing Solr image..."
    docker push $SOLR_REPO:latest
fi

echo ""
echo "Done! Images pushed to:"
echo "  ✓ $API_REPO:latest"
echo "  ✓ $WEB_REPO:latest"
if docker images | grep -q "$SOLR_API_REPO:latest"; then
    echo "  ✓ $SOLR_API_REPO:latest"
fi
if docker images | grep -q "$SOLR_REPO:latest"; then
    echo "  ✓ $SOLR_REPO:latest"
fi

