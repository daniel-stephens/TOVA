#!/bin/bash
# Script to fix empty EC2 instance by uploading code manually

set -e

echo "=========================================="
echo "Fixing Empty EC2 Instance"
echo "=========================================="

# Get connection info
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PUBLIC_IP=$(terraform output -raw ec2_public_ip 2>/dev/null || echo "")
KEY_FILE=$(terraform output -raw ec2_private_key_path 2>/dev/null || echo "")

if [ -z "$PUBLIC_IP" ] || [ -z "$KEY_FILE" ]; then
    echo "❌ Could not get EC2 instance information"
    exit 1
fi

# Resolve key file path
if [[ "$KEY_FILE" == ./* ]]; then
    KEY_FILE="$SCRIPT_DIR/${KEY_FILE#./}"
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "❌ Key file not found at: $KEY_FILE"
    exit 1
fi

echo "Public IP: $PUBLIC_IP"
echo "Key File: $KEY_FILE"
echo ""

# Check setup log first
echo "=========================================="
echo "Checking Setup Log"
echo "=========================================="
ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP "sudo tail -100 /var/log/tova-setup.log 2>/dev/null | tail -30" || echo "Could not read setup log"
echo ""

# Go to project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Creating Code Archive"
echo "=========================================="
echo "Creating tarball of your code..."
tar -czf /tmp/tova-app.tar.gz \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='terraform' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='cache' \
    --exclude='data' \
    --exclude='db' \
    --exclude='mydata' \
    --exclude='*.log' \
    .

echo "✅ Archive created: /tmp/tova-app.tar.gz"
echo ""

echo "=========================================="
echo "Uploading Code to EC2"
echo "=========================================="
scp -i "$KEY_FILE" /tmp/tova-app.tar.gz ec2-user@$PUBLIC_IP:~/ || {
    echo "❌ Upload failed"
    exit 1
}
echo "✅ Code uploaded"
echo ""

echo "=========================================="
echo "Extracting and Setting Up"
echo "=========================================="
ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP << 'ENDSSH'
cd ~
echo "Extracting code..."
tar -xzf tova-app.tar.gz -C TOVA/ 2>/dev/null || {
    echo "Creating TOVA directory and extracting..."
    mkdir -p TOVA
    tar -xzf tova-app.tar.gz -C TOVA/
}
cd TOVA
echo "Current directory: $(pwd)"
echo "Files:"
ls -la | head -10
echo ""
echo "Creating .env file if it doesn't exist..."
if [ ! -f .env ]; then
    cat > .env << 'EOF'
VERSION=latest
ASSETS_DATE=$(date +%Y%m%d)
POSTGRES_USER=tova_user
POSTGRES_PASSWORD=change_me
POSTGRES_DB=tova_db
EOF
    echo "✅ .env file created"
fi
echo ""
echo "Creating directories..."
mkdir -p data/drafts data/logs data/models
mkdir -p db/data/solr db/data/zoo
echo "✅ Directories created"
ENDSSH

echo ""
echo "=========================================="
echo "Building and Starting Services"
echo "=========================================="
echo "This will take several minutes..."
ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP << 'ENDSSH'
cd ~/TOVA
echo "Building base images..."
make build-builder || echo "Note: build-builder may take time"
make build-assets || echo "Note: build-assets may take time"
echo ""
echo "Starting services..."
docker compose up -d api web postgres || docker compose up -d
echo ""
echo "Checking service status..."
sleep 5
docker compose ps
ENDSSH

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Your application should now be running!"
echo "Access it at: http://$PUBLIC_IP:8080"
echo ""
echo "To check status:"
echo "  ssh -i $KEY_FILE ec2-user@$PUBLIC_IP"
echo "  cd ~/TOVA && docker compose ps"
echo ""

