# AWS App Runner Deployment

App Runner is a simpler alternative to ECS that automatically handles:
- Container orchestration
- Auto-scaling
- Load balancing
- HTTPS/SSL certificates
- Multi-platform image support (no architecture issues!)

## Quick Start

### 1. Configure Variables

```bash
cd terraform/apprunner
cp ../terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
```

You'll need:
- `vpc_id` - Use an existing VPC or create one
- `private_subnet_ids` - At least 2 subnets in different AZs
- `database_url` - Will be auto-generated from RDS

### 2. Get VPC Information

If using an existing VPC:

```bash
# List your VPCs
aws ec2 describe-vpcs --query 'Vpcs[*].[VpcId,CidrBlock,Tags[?Key==`Name`].Value|[0]]' --output table

# Get subnets for a VPC
aws ec2 describe-subnets --filters "Name=vpc-id,Values=vpc-xxxxx" --query 'Subnets[*].[SubnetId,AvailabilityZone]' --output table
```

### 3. Update terraform.tfvars

```hcl
vpc_id = "vpc-xxxxx"
private_subnet_ids = ["subnet-xxxxx", "subnet-yyyyy"]
```

### 4. Deploy

```bash
terraform init
terraform plan
terraform apply
```

### 5. Build and Push Images

App Runner automatically handles multi-platform images, but you still need to push:

```bash
# From project root
make build-api
make build-web

# Push to ECR (App Runner will handle architecture)
cd terraform/scripts
./build-and-push.sh
```

### 6. Access Your Application

```bash
terraform output web_url
terraform output api_url
```

## Advantages Over ECS

1. **No Architecture Issues** - App Runner handles multi-platform automatically
2. **Automatic HTTPS** - Free SSL certificates
3. **Auto-scaling** - Built-in
4. **Simpler** - Less configuration
5. **Built-in Load Balancer** - No need to configure ALB

## Limitations

- **Solr/Zookeeper** - Need separate deployment (ECS or managed service)
- **Less Control** - Can't customize infrastructure as much
- **Cost** - Slightly more expensive than ECS at scale
- **VPC Connector** - Required for RDS access (adds complexity)

## Architecture

```
Internet
   ↓
App Runner (API) → VPC Connector → RDS
App Runner (Web) → VPC Connector → RDS
```

## Cost Estimate

- **API Service**: ~$0.007/vCPU-hour, ~$0.0008/GB-hour
- **Web Service**: ~$0.007/vCPU-hour, ~$0.0008/GB-hour
- **VPC Connector**: ~$0.01/hour
- **RDS**: Same as ECS

**Estimated Monthly Cost (Dev):**
- 2 App Runner services (1 vCPU, 2GB each): ~$50
- VPC Connector: ~$7
- RDS db.t3.micro: ~$15
- **Total**: ~$70-80/month

## Next Steps

1. Deploy App Runner services
2. For Solr, either:
   - Use AWS OpenSearch (managed Solr alternative)
   - Deploy Solr via ECS separately
   - Use external Solr service

