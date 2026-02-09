# Deployment Options Comparison

TOVA can be deployed to AWS using two different approaches. Choose based on your needs:

## Option 1: ECS Fargate (Recommended) ⭐

**Best for:**
- Production deployments
- Need for fine-grained control
- Serverless container management
- Complex networking requirements
- Auto-scaling needs

**Pros:**
- ✅ Full control over infrastructure
- ✅ Serverless (no EC2 management)
- ✅ Better for production workloads
- ✅ More flexible networking
- ✅ Better cost optimization options
- ✅ Easier to integrate with other AWS services

**Cons:**
- ❌ More complex setup
- ❌ Requires more AWS knowledge
- ❌ More configuration files

**Location:** `terraform/` directory

**Quick Start:**
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init
terraform apply
```

## Option 2: Elastic Beanstalk Multi-Container Docker

**Best for:**
- Quick deployments
- Less infrastructure management
- Standard web applications
- Development/staging environments

**Pros:**
- ✅ Simpler setup
- ✅ Less infrastructure to manage
- ✅ Built-in monitoring and logging
- ✅ Automatic platform updates
- ✅ Easy rollback

**Cons:**
- ❌ Less control over infrastructure
- ❌ EC2-based (need to manage instances)
- ❌ Less flexible than ECS
- ❌ Platform limitations

**Location:** `terraform/elasticbeanstalk/` directory

**Quick Start:**
```bash
cd terraform/elasticbeanstalk
# Create Dockerrun.aws.json
# Package application
zip -r app.zip Dockerrun.aws.json docker-compose.yaml
# Configure and apply
terraform init
terraform apply
```

## Architecture Comparison

### ECS Fargate Architecture
```
Internet
   ↓
Application Load Balancer (ALB)
   ↓
ECS Fargate Tasks (API, Web)
   ↓
RDS PostgreSQL (Private Subnet)
```

**Components:**
- VPC with public/private subnets
- ECS Fargate cluster
- Application Load Balancer
- RDS PostgreSQL
- ECR for container images
- CloudWatch for logging

### Elastic Beanstalk Architecture
```
Internet
   ↓
Elastic Beanstalk Environment
   ├── EC2 Instances (Multi-container Docker)
   └── Application Load Balancer
   ↓
RDS PostgreSQL (Separate)
```

**Components:**
- Elastic Beanstalk environment
- EC2 instances (managed by EB)
- Application Load Balancer (managed by EB)
- RDS PostgreSQL (separate)
- S3 for application versions

## Cost Comparison

### ECS Fargate
- **Compute**: ~$0.04/vCPU-hour, ~$0.004/GB-hour
- **ALB**: ~$0.0225/hour + data transfer
- **RDS**: Depends on instance type
- **Data Transfer**: Standard AWS rates

**Estimated Monthly Cost (Dev):**
- 2 Fargate tasks (0.5 vCPU, 1GB each): ~$30
- ALB: ~$16
- RDS db.t3.micro: ~$15
- **Total**: ~$60-70/month

### Elastic Beanstalk
- **EC2**: Depends on instance type (t3.small ~$15/month)
- **ALB**: Included in EB
- **RDS**: Same as ECS
- **Data Transfer**: Standard AWS rates

**Estimated Monthly Cost (Dev):**
- EC2 t3.small: ~$15
- RDS db.t3.micro: ~$15
- **Total**: ~$30-40/month

*Note: Costs vary by region and usage. ECS can be more cost-effective at scale with reserved capacity.*

## Migration Path

You can start with Elastic Beanstalk and migrate to ECS later:

1. **Phase 1**: Deploy to Elastic Beanstalk (quick start)
2. **Phase 2**: Migrate to ECS Fargate (when you need more control)
3. **Phase 3**: Optimize and scale (auto-scaling, spot instances, etc.)

## Recommendation

**For Production:** Use **ECS Fargate**
- Better control and flexibility
- More suitable for production workloads
- Easier to integrate with CI/CD
- Better monitoring and observability

**For Development/Testing:** Either option works
- Elastic Beanstalk is faster to set up
- ECS Fargate gives you production-like environment

## Next Steps

1. Choose your deployment option
2. Follow the Quick Start guide for your chosen option
3. Review the main README.md for detailed instructions
4. Set up CI/CD for automated deployments
5. Configure monitoring and alerts

