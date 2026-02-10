#!/bin/bash
# TOVA Deployment Script
# This script helps set up and deploy TOVA with Gunicorn and Nginx

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="${PROJECT_DIR}/ui"
VENV_DIR="${UI_DIR}/venv"
NGINX_CONF="${PROJECT_DIR}/nginx/tova.conf"
SYSTEMD_SERVICE="${PROJECT_DIR}/systemd/tova-ui.service"

echo -e "${GREEN}TOVA Deployment Script${NC}"
echo "========================"
echo ""

# Check if running as root for system operations
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}Note: Some operations require sudo. You may be prompted for your password.${NC}"
    SUDO="sudo"
else
    SUDO=""
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."
if ! command_exists python3; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

if ! command_exists nginx; then
    echo -e "${YELLOW}Warning: nginx is not installed. Install it with: sudo apt-get install nginx${NC}"
fi

if ! command_exists psql; then
    echo -e "${YELLOW}Warning: PostgreSQL client is not installed. Install it with: sudo apt-get install postgresql-client${NC}"
fi

echo -e "${GREEN}✓ Prerequisites checked${NC}"
echo ""

# Setup virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment and install dependencies
echo "Installing Python dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${UI_DIR}/requirements.txt"
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check for .env file
if [ ! -f "${UI_DIR}/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found in ${UI_DIR}${NC}"
    echo "Create a .env file with the following variables:"
    echo "  FLASK_SECRET_KEY=your-secret-key"
    echo "  DATABASE_URL=postgresql+psycopg2://user:password@host:port/dbname"
    echo "  POSTGRES_HOST=localhost"
    echo "  POSTGRES_PORT=5432"
    echo "  POSTGRES_USER=your_user"
    echo "  POSTGRES_PASSWORD=your_password"
    echo "  POSTGRES_DB=your_db"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Test Gunicorn
echo "Testing Gunicorn configuration..."
cd "$UI_DIR"
if gunicorn --check-config --config gunicorn_config.py app:server 2>/dev/null; then
    echo -e "${GREEN}✓ Gunicorn configuration is valid${NC}"
else
    echo -e "${RED}Error: Gunicorn configuration test failed${NC}"
    exit 1
fi
echo ""

# Setup Nginx (if running with sudo)
if [ -n "$SUDO" ] && command_exists nginx; then
    echo "Setting up Nginx..."
    read -p "Do you want to install the Nginx configuration? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Update paths in nginx config (basic replacement)
        echo "Please update the paths in ${NGINX_CONF} before installing"
        read -p "Have you updated the paths? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $SUDO cp "$NGINX_CONF" /etc/nginx/sites-available/tova
            if [ ! -L /etc/nginx/sites-enabled/tova ]; then
                $SUDO ln -s /etc/nginx/sites-available/tova /etc/nginx/sites-enabled/
            fi
            if $SUDO nginx -t; then
                echo -e "${GREEN}✓ Nginx configuration installed and tested${NC}"
                read -p "Do you want to reload Nginx? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    $SUDO systemctl reload nginx
                    echo -e "${GREEN}✓ Nginx reloaded${NC}"
                fi
            else
                echo -e "${RED}Error: Nginx configuration test failed${NC}"
            fi
        fi
    fi
    echo ""
fi

# Setup systemd service (if running with sudo)
if [ -n "$SUDO" ]; then
    echo "Setting up systemd service..."
    read -p "Do you want to install the systemd service? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Please update the paths and environment variables in ${SYSTEMD_SERVICE} before installing"
        read -p "Have you updated the service file? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $SUDO cp "$SYSTEMD_SERVICE" /etc/systemd/system/
            $SUDO systemctl daemon-reload
            $SUDO systemctl enable tova-ui
            echo -e "${GREEN}✓ Systemd service installed and enabled${NC}"
            read -p "Do you want to start the service now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                $SUDO systemctl start tova-ui
                sleep 2
                if $SUDO systemctl is-active --quiet tova-ui; then
                    echo -e "${GREEN}✓ Service started successfully${NC}"
                    echo "Check status with: sudo systemctl status tova-ui"
                else
                    echo -e "${RED}Error: Service failed to start${NC}"
                    echo "Check logs with: sudo journalctl -u tova-ui -n 50"
                fi
            fi
        fi
    fi
    echo ""
fi

echo -e "${GREEN}Deployment setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Update environment variables in .env file or systemd service"
echo "2. Update paths in nginx/tova.conf and systemd/tova-ui.service"
echo "3. Test the application: gunicorn --config ui/gunicorn_config.py app:server"
echo "4. Start the service: sudo systemctl start tova-ui"
echo "5. Check logs: sudo journalctl -u tova-ui -f"
echo ""
echo "For detailed instructions, see DEPLOYMENT.md"

