#!/bin/bash
# Fix Docker image architecture for ECS Fargate (linux/amd64)
# This script rebuilds and pushes images with correct architecture

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

# Get version from .env
VERSION=$(grep "^VERSION=" ../.env 2>/dev/null | cut -d'=' -f2 || echo "latest")

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# ECR URLs
API_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${PROJECT_NAME}-api"
WEB_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${PROJECT_NAME}-web"

echo ""
echo "Building images for linux/amd64 (required for ECS Fargate)..."
echo "This will take 10-20 minutes due to emulation on Apple Silicon"
echo ""

# Build base images for linux/amd64
echo "1. Building base images..."
docker buildx build --platform linux/amd64 \
  --target builder \
  -t tova-builder:${VERSION} \
  -f ../docker/Dockerfile.base . --load

docker buildx build --platform linux/amd64 \
  --target assets \
  -t tova-assets:${VERSION} \
  -f ../docker/Dockerfile.base . --load

# Build API image
echo ""
echo "2. Building API image (this is the slowest part)..."
docker buildx build --platform linux/amd64 \
  --build-arg BUILDER_IMAGE=tova-builder:${VERSION} \
  --build-arg ASSETS_IMAGE=tova-assets:${VERSION} \
  -t ${API_REPO}:latest \
  -f ../docker/Dockerfile.api . --load

# Build Web image
echo ""
echo "3. Building Web image..."
docker buildx build --platform linux/amd64 \
  --build-arg ASSETS_IMAGE=tova-assets:${VERSION} \
  -t ${WEB_REPO}:latest \
  -f ../docker/Dockerfile.ui . --load

# Push images
echo ""
echo "4. Pushing images to ECR..."
docker push ${API_REPO}:latest
docker push ${WEB_REPO}:latest

echo ""
echo "✓ Done! Images built and pushed for linux/amd64"
echo ""
echo "Next: Force ECS service update:"
echo "  cd ../.."
echo "  cd terraform"
echo "  CLUSTER=\$(terraform output -raw cluster_name)"
echo "  aws ecs update-service --cluster \$CLUSTER --service tova-dev-api --force-new-deployment"
echo "  aws ecs update-service --cluster \$CLUSTER --service tova-dev-web --force-new-deployment"

