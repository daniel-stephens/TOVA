#!/bin/bash
# Quick fix script to release unassociated EIPs and update db_username

set -e

echo "=== Fixing Deployment Issues ==="
echo ""

# 1. Release unassociated EIPs
echo "1. Checking for unassociated Elastic IPs..."
UNASSOCIATED=$(aws ec2 describe-addresses \
  --filters "Name=domain,Values=vpc" \
  --query 'Addresses[?AssociationId==null].AllocationId' \
  --output text)

if [ -n "$UNASSOCIATED" ]; then
  echo "   Found unassociated EIPs:"
  for EIP in $UNASSOCIATED; do
    IP=$(aws ec2 describe-addresses --allocation-ids $EIP --query 'Addresses[0].PublicIp' --output text)
    echo "     - $EIP ($IP)"
  done
  
  read -p "   Release these EIPs? (y/N) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    for EIP in $UNASSOCIATED; do
      echo "   Releasing $EIP..."
      aws ec2 release-address --allocation-id $EIP
    done
    echo "   ✓ Released unassociated EIPs"
  else
    echo "   Skipped releasing EIPs"
  fi
else
  echo "   No unassociated EIPs found"
fi

echo ""

# 2. Update db_username in terraform.tfvars
echo "2. Checking terraform.tfvars for db_username..."
if [ -f "terraform.tfvars" ]; then
  if grep -q 'db_username.*=.*"admin"' terraform.tfvars || grep -q "db_username.*=.*admin" terraform.tfvars; then
    echo "   Found 'admin' username (reserved word)"
    read -p "   Update to 'tova_admin'? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      # Update using sed (works for both quoted and unquoted)
      sed -i.bak 's/db_username.*=.*admin/db_username = "tova_admin"/' terraform.tfvars
      echo "   ✓ Updated db_username to 'tova_admin'"
      echo "   (Backup saved as terraform.tfvars.bak)"
    else
      echo "   Please manually update db_username in terraform.tfvars"
    fi
  else
    echo "   db_username looks good"
  fi
else
  echo "   terraform.tfvars not found - will use default 'tova_admin'"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. If you released EIPs, you can now run: terraform apply"
echo "  2. If you still hit EIP limit, see RELEASE_EIPS.md for more options"

