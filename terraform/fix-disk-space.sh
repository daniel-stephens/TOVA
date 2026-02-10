#!/bin/bash
# Script to fix disk space issues on EC2 instance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PUBLIC_IP=$(terraform output -raw ec2_public_ip 2>/dev/null || echo "")
KEY_FILE=$(terraform output -raw ec2_private_key_path 2>/dev/null || echo "")

if [ -z "$PUBLIC_IP" ] || [ -z "$KEY_FILE" ]; then
    echo "❌ Could not get EC2 instance information"
    exit 1
fi

if [[ "$KEY_FILE" == ./* ]]; then
    KEY_FILE="$SCRIPT_DIR/${KEY_FILE#./}"
fi

echo "Connecting to EC2 instance to check and fix disk space..."
echo "Public IP: $PUBLIC_IP"
echo ""

ssh -i "$KEY_FILE" ubuntu@$PUBLIC_IP << 'EOF'
echo "=========================================="
echo "Current Disk Usage"
echo "=========================================="
df -h
echo ""

echo "=========================================="
echo "Docker Disk Usage"
echo "=========================================="
sudo docker system df
echo ""

echo "=========================================="
echo "Cleaning up Docker (this may take a while)..."
echo "=========================================="

# Stop all containers
echo "Stopping containers..."
sudo docker stop $(sudo docker ps -aq) 2>/dev/null || echo "No containers to stop"

# Remove all containers
echo "Removing containers..."
sudo docker rm $(sudo docker ps -aq) 2>/dev/null || echo "No containers to remove"

# Remove unused images
echo "Removing unused images..."
sudo docker image prune -af

# Remove build cache
echo "Removing build cache..."
sudo docker builder prune -af

# Remove unused volumes
echo "Removing unused volumes..."
sudo docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
sudo docker network prune -f

echo ""
echo "=========================================="
echo "Disk Usage After Cleanup"
echo "=========================================="
df -h
echo ""

echo "=========================================="
echo "Docker Disk Usage After Cleanup"
echo "=========================================="
sudo docker system df
echo ""

echo "=========================================="
echo "Cleaning up system packages..."
echo "=========================================="
if command -v apt-get &>/dev/null; then
    sudo apt-get clean
    sudo apt-get autoremove -y
elif command -v dnf &>/dev/null; then
    sudo dnf clean all
fi

echo ""
echo "=========================================="
echo "Final Disk Usage"
echo "=========================================="
df -h
echo ""

echo "✅ Cleanup complete! You can now retry the build."
echo ""
echo "To retry building, run:"
echo "  cd ~/TOVA"
echo "  sudo docker compose build"
echo "  sudo docker compose up -d"
EOF

