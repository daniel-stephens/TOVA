# How to Delete Terraform Resources

## Issue: ECR Repositories Can't Be Deleted

ECR repositories can't be deleted if they contain images. You have two options:

### Option 1: Delete Images First (Recommended)

```bash
# List images in repositories
aws ecr list-images --repository-name tova-api
aws ecr list-images --repository-name tova-web

# Delete all images
aws ecr batch-delete-image --repository-name tova-api \
  --image-ids imageTag=latest
aws ecr batch-delete-image --repository-name tova-web \
  --image-ids imageTag=latest

# Or delete all images
aws ecr list-images --repository-name tova-api --query 'imageIds[*]' --output json | \
  aws ecr batch-delete-image --repository-name tova-api --image-ids file:///dev/stdin
```

Then run:
```bash
terraform destroy
```

### Option 2: Enable Force Delete

Add to `terraform.tfvars`:

```hcl
force_delete_ecr = true
```

Then:
```bash
terraform apply  # Update the resources
terraform destroy  # Now it will delete repositories even with images
```

**Warning**: This will delete ALL images in the repositories!

### Option 3: Keep ECR Repositories

If you want to keep the images for future use, you can:

1. **Import existing repositories** (if recreating):
   ```bash
   terraform import module.ecr.aws_ecr_repository.repos["api"] tova-api
   terraform import module.ecr.aws_ecr_repository.repos["web"] tova-web
   ```

2. **Or manually delete** only the resources you want:
   ```bash
   # Target specific resources to destroy
   terraform destroy -target=module.ecs
   terraform destroy -target=module.rds
   # Keep ECR repositories
   ```

## Quick Delete Script

```bash
#!/bin/bash
# Delete all images from ECR repositories

REPOS=("tova-api" "tova-web" "tova-solr-api" "tova-solr" "tova-base")

for repo in "${REPOS[@]}"; do
  echo "Deleting images from $repo..."
  IMAGE_IDS=$(aws ecr list-images --repository-name $repo --query 'imageIds[*]' --output json)
  if [ "$IMAGE_IDS" != "[]" ]; then
    echo "$IMAGE_IDS" | aws ecr batch-delete-image --repository-name $repo --image-ids file:///dev/stdin
    echo "✓ Deleted images from $repo"
  else
    echo "  No images in $repo"
  fi
done

echo ""
echo "Now you can run: terraform destroy"
```

Save as `delete-ecr-images.sh`, make executable, and run:
```bash
chmod +x delete-ecr-images.sh
./delete-ecr-images.sh
```

## Complete Cleanup

To delete everything:

```bash
# 1. Delete ECR images
./delete-ecr-images.sh

# 2. Destroy infrastructure
terraform destroy

# 3. (Optional) Manually delete ECR repositories if needed
aws ecr delete-repository --repository-name tova-api --force
aws ecr delete-repository --repository-name tova-web --force
```

