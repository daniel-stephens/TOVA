#!/bin/bash
# Script to check EC2 instance status and diagnose issues

set -e

echo "=========================================="
echo "EC2 Instance Status Check"
echo "=========================================="

# Get outputs (handle being run from terraform/ or root directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/terraform.tfstate" ] || [ -f "$SCRIPT_DIR/terraform.tfvars" ]; then
    # We're in terraform directory
    TERRAFORM_DIR="$SCRIPT_DIR"
else
    # We're in root, terraform is subdirectory
    TERRAFORM_DIR="$SCRIPT_DIR/terraform"
fi

cd "$TERRAFORM_DIR" || {
    echo "❌ Cannot find terraform directory"
    exit 1
}

PUBLIC_IP=$(terraform output -raw ec2_public_ip 2>/dev/null || echo "")
KEY_FILE=$(terraform output -raw ec2_private_key_path 2>/dev/null || echo "")

if [ -z "$PUBLIC_IP" ] || [ -z "$KEY_FILE" ]; then
    echo "❌ Could not get EC2 instance information"
    echo "Make sure you've run: terraform apply"
    echo "And that deployment_type = 'ec2' in terraform.tfvars"
    exit 1
fi

# Resolve key file path relative to terraform directory
if [[ "$KEY_FILE" == ./* ]]; then
    KEY_FILE="$TERRAFORM_DIR/${KEY_FILE#./}"
fi

echo "Public IP: $PUBLIC_IP"
echo "Key File: $KEY_FILE"
echo ""

# Check if key file exists (handle relative paths)
if [[ "$KEY_FILE" == ./* ]]; then
    KEY_FILE="$TERRAFORM_DIR/${KEY_FILE#./}"
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "❌ Key file not found at: $KEY_FILE"
    echo "Expected location: $TERRAFORM_DIR/tova-dev-key.pem"
    exit 1
fi

echo "=========================================="
echo "SSH Connection Test"
echo "=========================================="

# Test SSH connection
echo "Testing SSH connection..."
ssh -i "$KEY_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP "echo 'SSH connection successful!'" 2>&1 || {
    echo "❌ Cannot connect to EC2 instance"
    echo "This might mean:"
    echo "  1. Instance is still starting (wait 2-3 minutes)"
    echo "  2. Security group doesn't allow SSH from your IP"
    echo "  3. Instance failed to start"
    exit 1
}

echo "✅ SSH connection successful!"
echo ""

echo "=========================================="
echo "Checking Setup Status"
echo "=========================================="

# Check setup log
echo "Checking setup log..."
ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP "sudo tail -50 /var/log/tova-setup.log 2>/dev/null || echo 'Setup log not found or empty'" || true
echo ""

echo "=========================================="
echo "Checking Application Directory"
echo "=========================================="

# Check if TOVA directory exists and has content
ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP << 'EOF'
echo "Checking /home/ec2-user/TOVA directory..."
if [ -d "/home/ec2-user/TOVA" ]; then
    echo "✅ Directory exists"
    echo "Contents:"
    ls -la /home/ec2-user/TOVA | head -20
    echo ""
    echo "Checking for key files:"
    [ -f "/home/ec2-user/TOVA/docker-compose.yaml" ] && echo "✅ docker-compose.yaml exists" || echo "❌ docker-compose.yaml missing"
    [ -f "/home/ec2-user/TOVA/Makefile" ] && echo "✅ Makefile exists" || echo "❌ Makefile missing"
    [ -d "/home/ec2-user/TOVA/ui" ] && echo "✅ ui/ directory exists" || echo "❌ ui/ directory missing"
else
    echo "❌ Directory does not exist!"
fi
EOF

echo ""
echo "=========================================="
echo "Checking Docker Status"
echo "=========================================="

ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP << 'EOF'
echo "Docker version:"
docker --version 2>/dev/null || echo "❌ Docker not installed"
echo ""
echo "Docker Compose version:"
(docker compose version 2>/dev/null || docker-compose --version 2>/dev/null) || echo "❌ Docker Compose not installed"
echo ""
echo "Running containers:"
docker ps 2>/dev/null || echo "No containers running"
EOF

echo ""
echo "=========================================="
echo "Checking Services"
echo "=========================================="

ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP << 'EOF'
if [ -d "/home/ec2-user/TOVA" ]; then
    cd /home/ec2-user/TOVA
    if docker compose version &>/dev/null; then DC="docker compose"; else DC="docker-compose"; fi
    echo "Docker Compose services:"
    $DC ps 2>/dev/null || echo "No services defined or docker compose not working"
else
    echo "Cannot check services - TOVA directory doesn't exist"
fi
EOF

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "If the directory is empty, you can:"
echo "1. Check if repo clone failed (see setup log above)"
echo "2. Manually upload code (see EC2_DEPLOYMENT.md)"
echo "3. SSH into instance: ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo ""

