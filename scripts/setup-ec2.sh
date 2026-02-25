#!/bin/bash

# TOVA EC2 Setup Script
# This script automates the setup of TOVA on a fresh EC2 instance
# Run this script on your EC2 instance after connecting via SSH

set -e  # Exit on error

echo "=========================================="
echo "TOVA EC2 Setup Script"
echo "=========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS. Exiting."
    exit 1
fi

echo "Detected OS: $OS"

# Function to install Docker (Amazon Linux 2023)
install_docker_amazon() {
    echo "Installing Docker for Amazon Linux..."
    sudo dnf install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ec2-user
    echo "Docker installed successfully!"
}

# Function to install Docker (Ubuntu)
install_docker_ubuntu() {
    echo "Installing Docker for Ubuntu..."
    sudo apt update
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ubuntu
    echo "Docker installed successfully!"
}

# Function to install Docker Compose
install_docker_compose() {
    echo "Installing Docker Compose..."
    if [ -f /usr/local/bin/docker-compose ]; then
        echo "Docker Compose already installed."
    else
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        echo "Docker Compose installed successfully!"
    fi
    (docker compose --version 2>/dev/null || docker-compose --version) || true
}

# Function to install Git
install_git() {
    echo "Installing Git..."
    if command -v git &> /dev/null; then
        echo "Git already installed."
    else
        if [ "$OS" = "amzn" ]; then
            sudo dnf install -y git
        else
            sudo apt install -y git
        fi
        echo "Git installed successfully!"
    fi
}

# Function to create .env file
create_env_file() {
    echo "Creating .env file..."
    if [ -f .env ]; then
        echo ".env file already exists. Skipping..."
    else
        cat > .env << EOF
VERSION=latest
ASSETS_DATE=$(date +%Y%m%d)
POSTGRES_USER=tova_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=tova_db
OPENAI_API_KEY=
FLASK_SECRET_KEY=$(openssl rand -base64 32)
EOF
        echo ".env file created!"
        echo "⚠️  IMPORTANT: Edit .env and set OPENAI_API_KEY if needed"
    fi
}

# Function to create directories
create_directories() {
    echo "Creating data directories..."
    mkdir -p data/drafts data/logs data/models
    mkdir -p db/data/solr db/data/zoo
    sudo chown -R $USER:$USER data db 2>/dev/null || true
    echo "Directories created!"
}

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Docker
    if command -v docker &> /dev/null; then
        echo "✅ Docker is installed"
        docker --version
    else
        echo "❌ Docker is not installed"
        return 1
    fi
    
    # Check Docker Compose (try plugin first, then standalone)
    if docker compose version &> /dev/null; then
        echo "✅ Docker Compose is installed (plugin)"
        docker compose version
    elif docker-compose version &> /dev/null; then
        echo "✅ Docker Compose is installed (standalone)"
        docker-compose version
    else
        echo "❌ Docker Compose is not installed"
        return 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        echo "✅ Git is installed"
    else
        echo "❌ Git is not installed"
        return 1
    fi
    
    echo "All prerequisites met!"
}

# Main installation
main() {
    echo ""
    echo "Step 1: Installing Docker..."
    if [ "$OS" = "amzn" ]; then
        install_docker_amazon
    elif [ "$OS" = "ubuntu" ]; then
        install_docker_ubuntu
    else
        echo "Unsupported OS. Please install Docker manually."
        exit 1
    fi
    
    echo ""
    echo "Step 2: Installing Docker Compose..."
    install_docker_compose
    
    echo ""
    echo "Step 3: Installing Git..."
    install_git
    
    echo ""
    echo "Step 4: Creating .env file..."
    create_env_file
    
    echo ""
    echo "Step 5: Creating directories..."
    create_directories
    
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Log out and log back in (for Docker group to take effect)"
    echo "2. Edit .env file and set OPENAI_API_KEY if needed:"
    echo "   nano .env"
    echo "3. Build base images:"
    echo "   make build-builder"
    echo "   make build-assets"
    echo "4. Start services:"
    echo "   docker compose up -d   # or: docker-compose up -d"
    echo "5. Check status:"
    echo "   docker compose ps      # or: docker-compose ps"
    echo "6. View logs:"
    echo "   docker compose logs -f # or: docker-compose logs -f"
    echo ""
    echo "⚠️  IMPORTANT: You need to log out and log back in for Docker group changes to take effect!"
    echo ""
}

# Run main function
main

