#!/bin/bash
# Delete all images from ECR repositories to allow terraform destroy

set -e

PROJECT_NAME="${PROJECT_NAME:-tova}"
REPOS=("api" "web" "solr-api" "solr" "base")

echo "Deleting images from ECR repositories..."
echo "Project: $PROJECT_NAME"
echo ""

for repo in "${REPOS[@]}"; do
  REPO_NAME="${PROJECT_NAME}-${repo}"
  echo "Checking $REPO_NAME..."
  
  # Check if repository exists
  if aws ecr describe-repositories --repository-names "$REPO_NAME" &>/dev/null; then
    # Get image IDs
    IMAGE_IDS=$(aws ecr list-images --repository-name "$REPO_NAME" --query 'imageIds[*]' --output json 2>/dev/null || echo "[]")
    
    if [ "$IMAGE_IDS" != "[]" ] && [ "$IMAGE_IDS" != "null" ] && [ -n "$IMAGE_IDS" ]; then
      echo "  Found images in $REPO_NAME, deleting..."
      echo "$IMAGE_IDS" | aws ecr batch-delete-image --repository-name "$REPO_NAME" --image-ids file:///dev/stdin 2>/dev/null || {
        # If batch delete fails, try deleting by tag
        echo "  Trying to delete by tag..."
        aws ecr list-images --repository-name "$REPO_NAME" --query 'imageIds[*]' --output text | \
          while read -r imageId; do
            if [ -n "$imageId" ]; then
              aws ecr batch-delete-image --repository-name "$REPO_NAME" --image-ids "$imageId" 2>/dev/null || true
            fi
          done
      }
      echo "  ✓ Deleted images from $REPO_NAME"
    else
      echo "  No images in $REPO_NAME"
    fi
  else
    echo "  Repository $REPO_NAME does not exist, skipping..."
  fi
done

echo ""
echo "✓ Done! You can now run: terraform destroy"
