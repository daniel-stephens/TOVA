# TOVA Terraform Deployment

This directory contains Terraform configurations for deploying TOVA to AWS using either:

1. **ECS Fargate** (Recommended for production) - More flexible, serverless containers
2. **Elastic Beanstalk Multi-Container Docker** (Simpler) - Easier setup, less control

## Prerequisites

1. **AWS CLI** installed and configured
2. **Terraform** >= 1.0 installed
3. **Docker** installed (for building and pushing images)
4. **AWS Account** with appropriate permissions

## Quick Start - ECS Fargate (Recommended)

### 1. Configure Variables

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Build and Push Docker Images to ECR

Before deploying, you need to build and push your Docker images to ECR:

```bash
# Get ECR login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build base images first (if needed)
make build-builder
make build-assets

# Build application images
make build-api
make build-web

# Tag and push to ECR (replace with your ECR URLs from terraform output)
docker tag tova-api:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/tova-api:latest
docker tag tova-web:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/tova-web:latest

docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/tova-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/tova-web:latest
```

Or use the provided script:

```bash
./scripts/build-and-push.sh
```

### 4. Plan and Apply

```bash
terraform plan
terraform apply
```

### 5. Access Your Application

After deployment, get the URLs:

```bash
terraform output web_url
terraform output api_url
```

## Elastic Beanstalk Deployment (Alternative)

Elastic Beanstalk is simpler but less flexible. It's good for:
- Quick deployments
- Less infrastructure management
- Standard web applications

### Setup

1. Navigate to the Elastic Beanstalk directory:

```bash
cd terraform/elasticbeanstalk
```

2. Configure variables (you'll need VPC and subnets created first, or use the VPC module from the main terraform)

3. Create a `Dockerrun.aws.json` file based on `Dockerrun.aws.json.example`

4. Package your application:

```bash
zip -r tova-eb-app.zip Dockerrun.aws.json docker-compose.yaml
```

5. Upload to S3 and deploy:

```bash
terraform init
terraform plan
terraform apply
```

## Architecture

### ECS Fargate Deployment

- **VPC**: Custom VPC with public and private subnets across multiple AZs
- **ECS Cluster**: Fargate cluster running containers
- **Application Load Balancer**: Routes traffic to API and Web services
- **RDS PostgreSQL**: Managed database in private subnets
- **ECR**: Container registries for Docker images
- **CloudWatch**: Logging and monitoring

### Services

- **API Service**: FastAPI application (port 8000)
- **Web Service**: Flask UI (port 8080)
- **Solr API Service**: Python API wrapper for Solr (port 8001)
- **Solr Service**: Apache Solr search engine (port 8983)
- **Zookeeper Service**: Coordination service for Solr (ports 2181, 8080)
- **Database**: PostgreSQL 15.4 (RDS)
- **EFS**: Elastic File System for persistent Solr/Zookeeper data

## Important Notes

### Database Password

**IMPORTANT**: Change the default database password in `terraform.tfvars` before deploying to production!

### SSL/HTTPS

To enable HTTPS:

1. Request an ACM certificate in AWS Certificate Manager
2. Add the certificate ARN to `terraform.tfvars`:
   ```hcl
   domain_name         = "tova.example.com"
   alb_certificate_arn = "arn:aws:acm:us-east-1:..."
   ```

### Secrets Management

For production, consider using AWS Secrets Manager for sensitive values like:
- Database passwords
- API keys
- Other secrets

Update the ECS task definitions to reference secrets from Secrets Manager.

### Solr Deployment

Solr is now fully integrated as ECS services:
- **Zookeeper**: ECS Fargate service for Solr coordination
- **Solr**: ECS Fargate service running Apache Solr
- **Solr API**: ECS Fargate service providing Python API wrapper
- **EFS**: Persistent storage for Solr indexes and Zookeeper data
- **Service Discovery**: Internal DNS for service-to-service communication

All Solr services are deployed in private subnets and communicate via AWS Service Discovery.

### Scaling

Default configuration uses:
- 1 task per service (API and Web)
- Auto-scaling can be configured in ECS service definitions
- RDS can be scaled vertically or use read replicas

### Cost Optimization

- Use `db.t3.micro` for development (free tier eligible)
- Consider Reserved Instances for production RDS
- Use Spot instances for non-critical workloads
- Enable CloudWatch log retention policies

## Troubleshooting

### ECS Tasks Not Starting

1. Check CloudWatch logs:
   ```bash
   aws logs tail /ecs/tova-dev-api --follow
   ```

2. Check ECS service events:
   ```bash
   aws ecs describe-services --cluster tova-dev-cluster --services tova-dev-api
   ```

3. Verify security groups allow traffic between ALB and ECS tasks

### Database Connection Issues

1. Verify RDS security group allows traffic from ECS security group
2. Check database endpoint is correct
3. Verify credentials in task definition

### Image Pull Errors

1. Ensure images are pushed to ECR
2. Verify ECR repository URLs in task definitions
3. Check IAM permissions for ECS task execution role

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete all resources including the database. Make sure you have backups!

## CI/CD Integration

Consider setting up CI/CD pipelines to:
1. Build Docker images on code changes
2. Push to ECR automatically
3. Update ECS services with new images
4. Run tests before deployment

Example GitHub Actions workflow can be added to `.github/workflows/`.

## Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Elastic Beanstalk Multi-Container Docker](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create_deploy_docker_ecs.html)

