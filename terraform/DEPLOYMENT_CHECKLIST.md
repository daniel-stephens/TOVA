# TOVA AWS Deployment Checklist

Follow these steps in order to deploy TOVA to AWS using Terraform.

## Prerequisites Check

- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Terraform >= 1.0 installed (`terraform version`)
- [ ] Docker installed and running
- [ ] AWS account with appropriate permissions
- [ ] `.env` file exists in project root with `VERSION` and `ASSETS_DATE`

## Step 1: Configure Terraform Variables

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:
- Set `aws_region` (e.g., `us-east-1`)
- Set `project_name` (default: `tova`)
- Set `environment` (e.g., `dev`, `staging`, `prod`)
- **IMPORTANT**: Set a strong `db_password`
- Optionally set `openai_api_key` if using OpenAI
- Optionally set `domain_name` and `alb_certificate_arn` for HTTPS

## Step 2: Initialize Terraform

```bash
terraform init
```

This downloads the AWS provider and initializes the backend.

## Step 3: Review Deployment Plan

```bash
terraform plan
```

Review what will be created:
- VPC and networking
- ECR repositories
- RDS database
- EFS file system
- ECS cluster and services
- Application Load Balancer

**Expected resources:**
- ~30-40 resources will be created
- Estimated time: 10-15 minutes

## Step 4: Deploy Infrastructure

```bash
terraform apply
```

Type `yes` when prompted. This will take about 10-15 minutes.

**What gets created:**
- VPC with public/private subnets
- ECR repositories (api, web, solr-api, solr, base)
- RDS PostgreSQL database
- EFS for persistent storage
- ECS Fargate cluster
- Application Load Balancer
- Security groups and IAM roles

## Step 5: Get ECR Repository URLs

After deployment completes, get your ECR URLs:

```bash
terraform output ecr_repository_urls
```

Save these URLs - you'll need them for pushing images.

## Step 6: Build Docker Images

### Option A: Build All Images (Recommended)

```bash
# From project root
make build-builder
make build-assets
make build-api
make build-web
make build-solr-api
make build-solr
```

### Option B: Use Build Script

```bash
# Build Solr images
./terraform/scripts/build-solr.sh

# Then build and push all images
cd terraform/scripts
./build-and-push.sh
```

## Step 7: Push Images to ECR

### Login to ECR

```bash
# Get your AWS account ID and region
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

# Login
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
```

### Tag and Push Images

```bash
# Get ECR URLs from terraform output
API_REPO=$(terraform output -json ecr_repository_urls | jq -r '.["api"]')
WEB_REPO=$(terraform output -json ecr_repository_urls | jq -r '.["web"]')
SOLR_API_REPO=$(terraform output -json ecr_repository_urls | jq -r '.["solr-api"]')
SOLR_REPO=$(terraform output -json ecr_repository_urls | jq -r '.["solr"]')

# Get version from .env
VERSION=$(grep "^VERSION=" ../.env | cut -d'=' -f2 || echo "latest")

# Tag images
docker tag tova-api:${VERSION} ${API_REPO}:latest
docker tag tova-web:${VERSION} ${WEB_REPO}:latest
docker tag tova-solr-api:${VERSION} ${SOLR_API_REPO}:latest
docker tag tova-solr:${VERSION} ${SOLR_REPO}:latest

# Push images
docker push ${API_REPO}:latest
docker push ${WEB_REPO}:latest
docker push ${SOLR_API_REPO}:latest
docker push ${SOLR_REPO}:latest
```

### Or Use the Build Script

```bash
cd terraform/scripts
./build-and-push.sh
```

## Step 8: Verify Services Are Running

```bash
# Get cluster name
CLUSTER=$(terraform output -raw cluster_name 2>/dev/null || echo "tova-dev-cluster")

# List services
aws ecs list-services --cluster $CLUSTER

# Check service status
aws ecs describe-services --cluster $CLUSTER \
  --services tova-dev-api tova-dev-web tova-dev-solr-api tova-dev-solr tova-dev-zookeeper
```

## Step 9: Access Your Application

```bash
# Get application URL
terraform output web_url
terraform output api_url
```

Open the web URL in your browser. The API should be accessible at `/api`.

## Step 10: Verify Everything Works

1. **Check Web UI**: Open the web URL - you should see the TOVA interface
2. **Check API**: Visit `/api/docs` for API documentation
3. **Check Health**: Visit `/health` endpoint
4. **Check Logs**: 
   ```bash
   aws logs tail /ecs/tova-dev-api --follow
   aws logs tail /ecs/tova-dev-web --follow
   ```

## Troubleshooting

### Services Not Starting

```bash
# Check service events
aws ecs describe-services --cluster $CLUSTER --services tova-dev-api

# Check task status
aws ecs list-tasks --cluster $CLUSTER --service-name tova-dev-api
aws ecs describe-tasks --cluster $CLUSTER --tasks <TASK_ID>
```

### Images Not Found

```bash
# Verify images are in ECR
aws ecr describe-images --repository-name tova-api
aws ecr describe-images --repository-name tova-solr-api
```

### Database Connection Issues

```bash
# Get database endpoint
terraform output rds_endpoint

# Check RDS status
aws rds describe-db-instances --db-instance-identifier tova-dev-db
```

### Can't Access Application

1. Check ALB target group health in AWS Console
2. Verify security groups allow traffic
3. Check CloudWatch logs for errors

## Next Steps

- [ ] Set up HTTPS with ACM certificate
- [ ] Configure custom domain
- [ ] Set up CI/CD pipeline
- [ ] Configure auto-scaling
- [ ] Set up monitoring and alerts
- [ ] Review and optimize costs

## Cleanup (When Done Testing)

```bash
terraform destroy
```

**Warning**: This deletes everything including the database!

## Getting Help

- Check `README.md` for detailed documentation
- Check `SOLR_SETUP.md` for Solr-specific issues
- Review CloudWatch logs for application errors
- Check AWS ECS console for service status

