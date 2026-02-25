#!/bin/bash
# Check for Docker-related errors in setup

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

echo "Checking Docker errors on EC2 instance..."
echo ""

ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP << 'EOF'
echo "=========================================="
echo "Docker Status"
echo "=========================================="
sudo systemctl status docker --no-pager | head -10
echo ""

echo "=========================================="
echo "Docker Errors in Setup Log"
echo "=========================================="
sudo grep -i "docker\|error\|failed\|warning" /var/log/tova-setup.log 2>/dev/null | tail -30 || echo "No Docker errors found in setup log"
echo ""

echo "=========================================="
echo "Startup Script Log"
echo "=========================================="
if [ -f /home/ec2-user/start-tova.sh.log ]; then
    tail -50 /home/ec2-user/start-tova.sh.log
else
    echo "Startup script log not found"
fi
echo ""

echo "=========================================="
echo "Docker Compose Status"
echo "=========================================="
if [ -d "/home/ec2-user/TOVA" ]; then
    cd /home/ec2-user/TOVA
    if docker compose version &>/dev/null; then DC="docker compose"; else DC="docker-compose"; fi
    $DC ps 2>&1 || echo "Docker compose command failed"
    echo ""
    echo "Docker compose logs (last 20 lines):"
    $DC logs --tail=20 2>&1 || echo "Could not get logs"
else
    echo "TOVA directory doesn't exist"
fi
echo ""

echo "=========================================="
echo "Docker System Info"
echo "=========================================="
docker info 2>&1 | head -20 || echo "Docker info failed"
echo ""

echo "=========================================="
echo "Running Containers"
echo "=========================================="
docker ps -a || echo "Docker ps failed"
EOF

