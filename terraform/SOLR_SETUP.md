# Solr Services Setup Guide

This guide explains how Solr services are configured in the ECS deployment.

## Architecture

The Solr stack consists of three ECS services:

1. **Zookeeper** - Coordination service for Solr clustering
2. **Solr** - Apache Solr search engine
3. **Solr API** - Python API wrapper for Solr

## Service Communication

Services communicate using AWS Service Discovery:

- **Namespace**: `tova.local` (or `${project_name}.local`)
- **Zookeeper**: `tova-dev-zookeeper.tova.local:2181`
- **Solr**: `tova-dev-solr.tova.local:8983`
- **Solr API**: `tova-dev-solr-api.tova.local:8001`

The API service connects to Solr API using the service discovery DNS name.

## Persistent Storage

Solr and Zookeeper data are stored on EFS (Elastic File System):

- **Solr data**: `/solr` access point (UID/GID 8983)
- **Zookeeper data**: `/zookeeper` access point (UID/GID 1000)

EFS provides:
- Persistent storage across container restarts
- Shared storage for multi-AZ deployments
- Automatic backups via EFS lifecycle policies

## Building Solr Images

Before deploying, you need to build and push Solr images:

### 1. Build Solr Images

**Option A: Using Makefile (Recommended)**

Ensure you have a `.env` file with `VERSION` set:

```bash
# From project root
# Build builder image first (if not already built)
make build-builder

# Build Solr images
make build-solr-api
make build-solr
```

**Option B: Using Build Script**

```bash
# From terraform/scripts directory
./build-solr.sh [version]

# Or from project root
./terraform/scripts/build-solr.sh
```

**Option C: Manual Build**

```bash
# From project root
# 1. Build builder image first
make build-builder

# 2. Build Solr API (replace ${VERSION} with your version from .env)
docker build \
  --build-arg BUILDER_IMAGE=tova-builder:${VERSION} \
  -t tova-solr-api:${VERSION} \
  -f docker/Dockerfile.solr_api .

# 3. Build Solr
docker build \
  -t tova-solr:${VERSION} \
  -f docker/Dockerfile.solr_db .
```

**Note**: The `BUILDER_IMAGE` build arg is required for Solr API. If you get an error about invalid reference format, ensure:
1. The builder image exists: `docker images | grep tova-builder`
2. You're passing the correct version: `--build-arg BUILDER_IMAGE=tova-builder:${VERSION}`

### 3. Push to ECR

After infrastructure is created, push images:

```bash
# Get ECR URLs
terraform output ecr_repository_urls

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag tova-solr:latest <ECR_SOLR_URL>:latest
docker tag tova-solr-api:latest <ECR_SOLR_API_URL>:latest

docker push <ECR_SOLR_URL>:latest
docker push <ECR_SOLR_API_URL>:latest
```

## Initialization

Solr configuration is initialized automatically when the Solr container starts. The Dockerfile includes:

- Configsets from `solr/config/configsets`
- Custom plugins from `solr/plugins/`

If you need to run initialization scripts (like `solr_config` from docker-compose), you can:

1. Add an init container to the Solr task definition
2. Run initialization as part of the Solr startup command
3. Use a separate one-time ECS task

## Configuration

### Solr Configuration

Solr configuration is baked into the Docker image. To update:

1. Modify files in `solr/config/configsets/`
2. Rebuild the Solr image
3. Push to ECR
4. Update ECS service to force new deployment

### Environment Variables

**Zookeeper:**
- `JVMFLAGS`: JVM flags (default: `-Djute.maxbuffer=50000000`)

**Solr:**
- `ZK_HOST`: Zookeeper connection string (auto-configured via service discovery)

**Solr API:**
- `SOLR_URL`: Solr connection URL (auto-configured via service discovery)

## Scaling

### Current Configuration

- **Zookeeper**: 1 task (single instance)
- **Solr**: 1 task (can be scaled for clustering)
- **Solr API**: 1 task (can be scaled horizontally)

### Scaling Solr

To scale Solr for clustering:

1. Increase `desired_count` in the Solr service
2. Ensure Zookeeper can handle multiple Solr nodes
3. Update Solr configuration for distributed search

### Scaling Solr API

Solr API can be scaled independently:

```bash
aws ecs update-service \
  --cluster tova-dev-cluster \
  --service tova-dev-solr-api \
  --desired-count 3
```

## Monitoring

### CloudWatch Logs

Each service has its own log group:

- `/ecs/tova-dev-zookeeper`
- `/ecs/tova-dev-solr`
- `/ecs/tova-dev-solr-api`

View logs:

```bash
aws logs tail /ecs/tova-dev-solr --follow
```

### Health Checks

Currently, health checks are not configured for Solr services. To add:

1. Add health check endpoint to Solr API
2. Configure health check in ECS service
3. Set up CloudWatch alarms

## Troubleshooting

### Solr Can't Connect to Zookeeper

1. Check Zookeeper service is running:
   ```bash
   aws ecs describe-services --cluster tova-dev-cluster --services tova-dev-zookeeper
   ```

2. Verify service discovery DNS:
   ```bash
   # From within an ECS task
   nslookup tova-dev-zookeeper.tova.local
   ```

3. Check security groups allow traffic on port 2181

### EFS Mount Issues

1. Verify EFS mount targets are in correct subnets
2. Check EFS security group allows NFS (port 2049)
3. Verify IAM role has EFS permissions
4. Check CloudWatch logs for mount errors

### Data Persistence Issues

1. Verify EFS access points are created:
   ```bash
   aws efs describe-access-points --file-system-id <EFS_ID>
   ```

2. Check EFS file system status:
   ```bash
   aws efs describe-file-systems
   ```

3. Verify data is being written to EFS (check EFS metrics in CloudWatch)

## Migration from Docker Compose

If migrating from local Docker Compose deployment:

1. **Data Migration**: Copy Solr indexes from local `db/data/solr` to EFS
2. **Configuration**: Ensure configsets are in the Docker image
3. **Environment Variables**: Update any hardcoded service names to use service discovery
4. **Volumes**: Replace volume mounts with EFS access points

## Cost Considerations

- **EFS**: ~$0.30/GB-month for storage, ~$0.01/GB for data transfer
- **ECS Tasks**: Same pricing as other services
- **Data Transfer**: Between services is free within VPC

For cost optimization:
- Use EFS Infrequent Access for old indexes
- Enable EFS lifecycle policies
- Consider EBS volumes for single-AZ deployments (cheaper but less resilient)

