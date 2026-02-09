# Troubleshooting "Temporarily Unavailable" Error

If you're seeing "503 Service Temporarily Unavailable" when accessing your site, follow these steps:

## Quick Diagnosis

```bash
cd terraform

# 1. Check service status
CLUSTER=$(terraform output -raw cluster_name)
aws ecs describe-services --cluster $CLUSTER --services tova-dev-api tova-dev-web \
  --query 'services[*].[serviceName,runningCount,desiredCount]' --output table

# 2. Check service events (why tasks aren't starting)
aws ecs describe-services --cluster $CLUSTER --services tova-dev-api \
  --query 'services[0].events[0:5]' --output table

# 3. Check stopped tasks (error messages)
aws ecs list-tasks --cluster $CLUSTER --service-name tova-dev-api \
  --desired-status STOPPED --max-items 1 \
  --query 'taskArns[0]' --output text | \
  xargs -I {} aws ecs describe-tasks --cluster $CLUSTER --tasks {} \
  --query 'tasks[0].[stoppedReason,containers[0].reason]' --output text
```

## Common Issues and Fixes

### Issue 1: No Docker Images in ECR

**Symptoms:**
- Tasks fail with "CannotPullContainerError"
- Service events show "failed to pull image"

**Fix:**
```bash
# Build and push images
cd ..
make build-builder
make build-assets
make build-api
make build-web

# Push to ECR
cd terraform/scripts
./build-and-push.sh
```

### Issue 2: Tasks Failing to Start

**Symptoms:**
- Tasks show as STOPPED
- Exit code is non-zero

**Check logs:**
```bash
# Check CloudWatch logs
aws logs tail /ecs/tova-dev-api --follow
aws logs tail /ecs/tova-dev-web --follow
```

**Common causes:**
- Database connection issues
- Missing environment variables
- Health check failures

### Issue 3: Health Check Failures

**Symptoms:**
- Tasks running but target group shows unhealthy
- ALB returns 503

**Check:**
```bash
# Get target group health
ALB_ARN=$(terraform output -raw alb_arn)
aws elbv2 describe-target-health \
  --target-group-arn $(aws elbv2 describe-target-groups \
    --load-balancer-arn $ALB_ARN \
    --query 'TargetGroups[?TargetGroupName==`tova-dev-api-tg`].TargetGroupArn' \
    --output text) \
  --query 'TargetHealthDescriptions[*].[Target.Id,TargetHealth.State,TargetHealth.Reason]' \
  --output table
```

**Fix:**
- Verify health check path exists (`/health` for API)
- Check security groups allow traffic from ALB
- Verify containers are listening on correct ports

### Issue 4: Database Connection Issues

**Symptoms:**
- Tasks start but immediately fail
- Logs show database connection errors

**Fix:**
```bash
# Verify database endpoint
terraform output rds_endpoint

# Check RDS security group allows traffic from ECS
# Check database credentials in task definition
```

### Issue 5: Missing Environment Variables

**Symptoms:**
- Application errors in logs
- Tasks crash on startup

**Fix:**
- Verify all required environment variables are set in task definitions
- Check `terraform.tfvars` has correct values

## Step-by-Step Recovery

1. **Check if images are pushed:**
   ```bash
   terraform output ecr_repository_urls
   aws ecr describe-images --repository-name tova-api
   ```

2. **If images missing, build and push:**
   ```bash
   cd ..
   make build-api build-web
   cd terraform/scripts
   ./build-and-push.sh
   ```

3. **Force service update:**
   ```bash
   CLUSTER=$(terraform output -raw cluster_name)
   aws ecs update-service --cluster $CLUSTER --service tova-dev-api --force-new-deployment
   aws ecs update-service --cluster $CLUSTER --service tova-dev-web --force-new-deployment
   ```

4. **Monitor service events:**
   ```bash
   watch -n 5 "aws ecs describe-services --cluster $CLUSTER --services tova-dev-api --query 'services[0].events[0:3]' --output table"
   ```

5. **Check logs once tasks are running:**
   ```bash
   aws logs tail /ecs/tova-dev-api --follow
   ```

## Getting Help

If issues persist:
1. Check CloudWatch logs for detailed error messages
2. Review ECS service events for deployment issues
3. Verify all infrastructure is created correctly: `terraform plan`
4. Check AWS Console for visual debugging

