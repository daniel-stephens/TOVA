# Fix: "Temporarily Unavailable" Error

## Problem
Your ECS services can't start because the Docker images haven't been pushed to ECR yet.

**Error:** `CannotPullContainerError: tova-api:latest: not found`

## Solution: Build and Push Docker Images

### Step 1: Get ECR Repository URLs

```bash
cd terraform
terraform output ecr_repository_urls
```

### Step 2: Login to ECR

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
```

### Step 3: Build Images

```bash
cd ..  # Go to project root

# Build base images
make build-builder
make build-assets

# Build application images
make build-api
make build-web
```

### Step 4: Tag and Push Images

```bash
# Get ECR URLs
API_REPO=$(cd terraform && terraform output -json ecr_repository_urls | python3 -c "import sys, json; print(json.load(sys.stdin)['api'])")
WEB_REPO=$(cd terraform && terraform output -json ecr_repository_urls | python3 -c "import sys, json; print(json.load(sys.stdin)['web'])")

# Get version from .env or use 'latest'
VERSION=$(grep "^VERSION=" .env 2>/dev/null | cut -d'=' -f2 || echo "latest")

# Tag images
docker tag tova-api:${VERSION} ${API_REPO}:latest 2>/dev/null || docker tag tova-api:latest ${API_REPO}:latest
docker tag tova-web:${VERSION} ${WEB_REPO}:latest 2>/dev/null || docker tag tova-web:latest ${WEB_REPO}:latest

# Push images
docker push ${API_REPO}:latest
docker push ${WEB_REPO}:latest
```

### Step 5: Force Service Update

```bash
cd terraform
CLUSTER=$(terraform output -raw cluster_name)

# Force ECS to pull new images
aws ecs update-service --cluster $CLUSTER --service tova-dev-api --force-new-deployment
aws ecs update-service --cluster $CLUSTER --service tova-dev-web --force-new-deployment
```

### Step 6: Monitor Deployment

```bash
# Watch service events
watch -n 5 "aws ecs describe-services --cluster $CLUSTER --services tova-dev-api --query 'services[0].events[0:3]' --output table"

# Or check status
aws ecs describe-services --cluster $CLUSTER --services tova-dev-api tova-dev-web \
  --query 'services[*].[serviceName,runningCount,desiredCount]' --output table
```

## Quick Script

Or use the provided script:

```bash
cd terraform/scripts
./build-and-push.sh
```

Then force service update:

```bash
cd ../..
cd terraform
CLUSTER=$(terraform output -raw cluster_name)
aws ecs update-service --cluster $CLUSTER --service tova-dev-api --force-new-deployment
aws ecs update-service --cluster $CLUSTER --service tova-dev-web --force-new-deployment
```

## Verify

After a few minutes, check:

```bash
# Services should show runningCount = 1
aws ecs describe-services --cluster $CLUSTER --services tova-dev-api tova-dev-web \
  --query 'services[*].[serviceName,runningCount,desiredCount]' --output table

# Get your site URL
terraform output web_url
```

Your site should now be accessible!

