#!/bin/bash
# Quick script to check what happened during automatic setup

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

echo "Checking setup log on EC2 instance..."
echo "Public IP: $PUBLIC_IP"
echo ""

ssh -i "$KEY_FILE" ec2-user@$PUBLIC_IP << 'EOF'
echo "=========================================="
echo "Setup Log (Last 100 lines)"
echo "=========================================="
sudo tail -100 /var/log/tova-setup.log 2>/dev/null || echo "Setup log not found"
echo ""
echo "=========================================="
echo "Checking for Git Clone Errors"
echo "=========================================="
sudo grep -i "clone\|git\|repo\|failed" /var/log/tova-setup.log 2>/dev/null | tail -20 || echo "No git-related errors found"
echo ""
echo "=========================================="
echo "Checking if TOVA directory exists"
echo "=========================================="
if [ -d "/home/ec2-user/TOVA" ]; then
    echo "✅ Directory exists"
    echo "File count: $(find /home/ec2-user/TOVA -type f 2>/dev/null | wc -l)"
    echo "Has docker-compose.yaml: $([ -f /home/ec2-user/TOVA/docker-compose.yaml ] && echo 'Yes' || echo 'No')"
else
    echo "❌ Directory does not exist"
fi
EOF

