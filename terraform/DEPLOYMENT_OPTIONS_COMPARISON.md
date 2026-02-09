# Deployment Options Comparison

You have several options for deploying TOVA. Here's a comparison:

## Option 1: AWS App Runner ⭐ (Simplest)

**Best for:** Quick deployment, less infrastructure management

**Pros:**
- ✅ Fully managed - AWS handles everything
- ✅ Automatic scaling
- ✅ Built-in load balancing
- ✅ Automatic HTTPS
- ✅ No architecture issues (handles multi-platform)
- ✅ Simpler configuration
- ✅ Automatic deployments from ECR

**Cons:**
- ❌ Less control over infrastructure
- ❌ More expensive at scale
- ❌ Limited customization
- ❌ Solr/Zookeeper need separate deployment (ECS or managed service)

**Location:** `terraform/apprunner/`

**Quick Start:**
```bash
cd terraform/apprunner
# Configure variables
terraform init
terraform apply
```

## Option 2: ECS Fargate (Current - Fix Architecture Issue)

**Best for:** Production, full control, cost optimization

**Pros:**
- ✅ Full control over infrastructure
- ✅ Cost-effective at scale
- ✅ Can deploy all services (including Solr)
- ✅ More flexible

**Cons:**
- ❌ More complex setup
- ❌ Need to build images for correct architecture (linux/amd64)
- ❌ More configuration

**Fix:** Build images with `--platform linux/amd64` flag

## Option 3: ECS EC2 Launch Type

**Best for:** Need more control, custom AMIs, GPU support

**Pros:**
- ✅ Full control over EC2 instances
- ✅ Can use custom AMIs
- ✅ GPU support
- ✅ More predictable costs

**Cons:**
- ❌ Need to manage EC2 instances
- ❌ More complex than Fargate
- ❌ Need to handle instance scaling

## Option 4: AWS Lightsail Containers

**Best for:** Simple deployments, low cost, small scale

**Pros:**
- ✅ Very simple
- ✅ Low cost
- ✅ Good for small deployments

**Cons:**
- ❌ Limited scalability
- ❌ Less features
- ❌ Regional limitations

## Recommendation

**For Quick Deployment:** Use **App Runner** - it's the simplest and handles architecture automatically.

**For Production:** Fix the ECS Fargate setup by building images correctly for linux/amd64.

## Quick Fix for Current ECS Setup

If you want to stick with ECS but fix the architecture issue:

```bash
# Build for linux/amd64
docker buildx build --platform linux/amd64 \
  --build-arg BUILDER_IMAGE=tova-builder:latest \
  --build-arg ASSETS_IMAGE=tova-assets:latest \
  -t 152114282041.dkr.ecr.us-east-1.amazonaws.com/tova-api:latest \
  -f docker/Dockerfile.api . --load

# Push
docker push 152114282041.dkr.ecr.us-east-1.amazonaws.com/tova-api:latest
```

Then update ECS services.

