#!/bin/bash
set -e

# Log everything to a file for debugging
exec > >(tee /var/log/tova-setup.log)
exec 2>&1

echo "=========================================="
echo "TOVA EC2 Setup Script"
echo "Started at: $(date)"
echo "=========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    echo "Cannot detect OS. Exiting."
    exit 1
fi

echo "Detected OS: $OS $OS_VERSION"

# Function to install Docker (Amazon Linux 2023)
install_docker_amazon() {
    echo "[$(date)] Installing Docker for Amazon Linux..."
    sudo dnf update -y
    sudo dnf install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ec2-user
    echo "[$(date)] Docker installed successfully!"
    docker --version
}

# Function to install Docker (Ubuntu)
install_docker_ubuntu() {
    echo "[$(date)] Installing Docker for Ubuntu..."
    sudo apt update -y
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ubuntu
    echo "[$(date)] Docker installed successfully!"
    docker --version
}

# Function to install Docker Compose
install_docker_compose() {
    echo "[$(date)] Installing Docker Compose..."
    if [ -f /usr/local/bin/docker-compose ]; then
        echo "Docker Compose already installed."
    else
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        echo "[$(date)] Docker Compose installed successfully!"
    fi
    docker-compose --version
}

# Function to install Git
install_git() {
    echo "[$(date)] Installing Git..."
    if command -v git &> /dev/null; then
        echo "Git already installed."
    else
        if [ "$OS" = "amzn" ]; then
            sudo dnf install -y git
        else
            sudo apt install -y git
        fi
        echo "[$(date)] Git installed successfully!"
    fi
    git --version
}

# Function to install additional tools
install_additional_tools() {
    echo "[$(date)] Installing additional tools..."
    if [ "$OS" = "amzn" ]; then
        sudo dnf install -y make curl wget unzip
    else
        sudo apt install -y make curl wget unzip
    fi
    echo "[$(date)] Additional tools installed!"
}

# Main installation
echo "[$(date)] Starting installation process..."

# Step 1: Install Docker
if [ "$OS" = "amzn" ]; then
    install_docker_amazon
elif [ "$OS" = "ubuntu" ]; then
    install_docker_ubuntu
else
    echo "Unsupported OS: $OS. Please install Docker manually."
    exit 1
fi

# Step 2: Install Docker Compose
install_docker_compose

# Step 3: Install Git
install_git

# Step 4: Install additional tools
install_additional_tools

# Step 5: Clone or prepare repository
echo "[$(date)] Setting up application directory..."
APP_DIR="/home/ec2-user/TOVA"

if [ -n "${repo_url}" ] && [ "${repo_url}" != "" ]; then
    echo "[$(date)] Cloning repository from ${repo_url} (branch: ${repo_branch})..."
    sudo -u ec2-user git clone -b ${repo_branch} ${repo_url} $APP_DIR || {
        echo "[$(date)] Failed to clone repository. Creating directory structure..."
        sudo -u ec2-user mkdir -p $APP_DIR
    }
else
    echo "[$(date)] No repository URL provided. Creating directory structure..."
    sudo -u ec2-user mkdir -p $APP_DIR
fi

cd $APP_DIR

# Step 6: Create .env file
echo "[$(date)] Creating .env file..."
sudo -u ec2-user cat > .env << EOF
VERSION=latest
ASSETS_DATE=$(date +%Y%m%d)
POSTGRES_USER=${postgres_user}
POSTGRES_PASSWORD=${postgres_password}
POSTGRES_DB=${postgres_db}
DATABASE_URL=postgresql://${postgres_user}:${postgres_password}@${postgres_host}:5432/${postgres_db}
OPENAI_API_KEY=${openai_api_key}
FLASK_SECRET_KEY=${flask_secret_key}
API_BASE_URL=http://localhost:8000
EOF

echo "[$(date)] .env file created!"

# Step 7: Create directories
echo "[$(date)] Creating data directories..."
sudo -u ec2-user mkdir -p data/drafts data/logs data/models
sudo -u ec2-user mkdir -p db/data/solr db/data/zoo
sudo chown -R ec2-user:ec2-user data db 2>/dev/null || sudo chown -R ubuntu:ubuntu data db 2>/dev/null || true
echo "[$(date)] Directories created!"

# Step 8: Create startup script
echo "[$(date)] Creating startup script..."
sudo -u ec2-user cat > /home/ec2-user/start-tova.sh << 'SCRIPT'
#!/bin/bash
set -e

cd /home/ec2-user/TOVA

echo "[$(date)] Starting TOVA services..."

# Wait a bit for Docker to be fully ready
sleep 5

# Build base images
echo "[$(date)] Building base images..."
make build-builder || echo "Warning: build-builder failed"
make build-assets || echo "Warning: build-assets failed"

# Start services
if [ "${enable_solr}" = "true" ]; then
    echo "[$(date)] Starting all services including Solr..."
    docker compose up -d api web postgres solr-api solr zoo solr_config || docker compose up -d
else
    echo "[$(date)] Starting services without Solr..."
    docker compose up -d api web postgres || docker compose up -d
fi

echo "[$(date)] Services started!"
echo "[$(date)] Check status with: docker compose ps"
echo "[$(date)] View logs with: docker compose logs -f"
SCRIPT

chmod +x /home/ec2-user/start-tova.sh
sudo chown ec2-user:ec2-user /home/ec2-user/start-tova.sh 2>/dev/null || sudo chown ubuntu:ubuntu /home/ec2-user/start-tova.sh 2>/dev/null || true

# Step 9: Create systemd service for auto-start
echo "[$(date)] Creating systemd service..."
sudo cat > /etc/systemd/system/tova.service << EOF
[Unit]
Description=TOVA Application
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/TOVA
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ec2-user
Group=ec2-user
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Try ubuntu user if ec2-user doesn't exist
if ! id "ec2-user" &>/dev/null; then
    sudo sed -i 's/ec2-user/ubuntu/g' /etc/systemd/system/tova.service
fi

sudo systemctl daemon-reload
sudo systemctl enable tova.service

# Step 10: Install CloudWatch agent (optional)
echo "[$(date)] Installation complete!"
echo "[$(date)] Setup log saved to: /var/log/tova-setup.log"

# Step 11: Run startup script in background (after a delay to ensure Docker is ready)
echo "[$(date)] Scheduling application startup..."
sudo -u ec2-user bash -c "sleep 30 && /home/ec2-user/start-tova.sh" &

echo "=========================================="
echo "TOVA EC2 Setup Complete!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Wait 2-3 minutes for services to start"
echo "2. Check status: docker compose ps"
echo "3. View logs: docker compose logs -f"
echo "4. Access Web UI: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
echo "5. Access API: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo ""
echo "Setup log: /var/log/tova-setup.log"

