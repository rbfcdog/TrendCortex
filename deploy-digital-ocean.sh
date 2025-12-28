#!/bin/bash

# TrendCortex Digital Ocean Deployment Script
# This script automates deployment to a Digital Ocean droplet

set -e

echo "=========================================="
echo "TrendCortex - Digital Ocean Deployment"
echo "=========================================="
echo ""

# Configuration
DROPLET_NAME="${DROPLET_NAME:-trendcortex-bot}"
DROPLET_SIZE="${DROPLET_SIZE:-s-2vcpu-4gb}"  # 2 vCPU, 4GB RAM
DROPLET_REGION="${DROPLET_REGION:-nyc1}"
DROPLET_IMAGE="${DROPLET_IMAGE:-docker-20-04}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    log_error "doctl CLI is not installed"
    echo ""
    echo "Please install doctl:"
    echo "  brew install doctl  # macOS"
    echo "  snap install doctl  # Linux"
    echo ""
    echo "Then authenticate:"
    echo "  doctl auth init"
    exit 1
fi

# Check if docker is installed locally
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed locally"
    exit 1
fi

# Step 1: Build Docker image locally
log_info "Building Docker image locally..."
docker build -t trendcortex:latest .

# Step 2: Check if config.json exists
if [ ! -f "config.json" ]; then
    log_error "config.json not found!"
    echo ""
    echo "Please create config.json with your WEEX API credentials:"
    echo "  cp config.example.json config.json"
    echo "  nano config.json"
    exit 1
fi

# Step 3: Check if droplet exists
log_info "Checking if droplet exists..."
if doctl compute droplet list --format Name --no-header | grep -q "^${DROPLET_NAME}$"; then
    log_info "Droplet '${DROPLET_NAME}' already exists"
    DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep "^${DROPLET_NAME}" | awk '{print $2}')
else
    # Step 4: Create droplet
    log_info "Creating Digital Ocean droplet..."
    log_info "  Name: ${DROPLET_NAME}"
    log_info "  Size: ${DROPLET_SIZE}"
    log_info "  Region: ${DROPLET_REGION}"
    log_info "  Image: ${DROPLET_IMAGE}"
    
    doctl compute droplet create ${DROPLET_NAME} \
        --size ${DROPLET_SIZE} \
        --image ${DROPLET_IMAGE} \
        --region ${DROPLET_REGION} \
        --ssh-keys $(doctl compute ssh-key list --format ID --no-header | head -n 1) \
        --wait
    
    # Get droplet IP
    DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep "^${DROPLET_NAME}" | awk '{print $2}')
    
    log_info "Droplet created with IP: ${DROPLET_IP}"
    log_warn "Waiting 30 seconds for droplet to be ready..."
    sleep 30
fi

log_info "Droplet IP: ${DROPLET_IP}"

# Step 5: Upload files to droplet
log_info "Uploading files to droplet..."

# Create deployment directory on droplet
ssh -o StrictHostKeyChecking=no root@${DROPLET_IP} "mkdir -p /opt/trendcortex"

# Upload necessary files
scp -o StrictHostKeyChecking=no -r \
    trendcortex \
    *.py \
    *.sh \
    requirements.txt \
    Dockerfile \
    docker-compose.yml \
    config.json \
    root@${DROPLET_IP}:/opt/trendcortex/

# Create directories
ssh root@${DROPLET_IP} "cd /opt/trendcortex && mkdir -p logs data models"

# Step 6: Install Docker on droplet (if needed)
log_info "Ensuring Docker is installed on droplet..."
ssh root@${DROPLET_IP} "docker --version || curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"

# Step 7: Install Docker Compose on droplet (if needed)
log_info "Ensuring Docker Compose is installed on droplet..."
ssh root@${DROPLET_IP} "docker-compose --version || (curl -L 'https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)' -o /usr/local/bin/docker-compose && chmod +x /usr/local/bin/docker-compose)"

# Step 8: Build and start containers on droplet
log_info "Building and starting TrendCortex bot..."
ssh root@${DROPLET_IP} "cd /opt/trendcortex && docker-compose down && docker-compose build && docker-compose up -d"

# Step 9: Wait and check status
log_info "Waiting for bot to start..."
sleep 10

ssh root@${DROPLET_IP} "cd /opt/trendcortex && docker-compose ps"

# Step 10: Show logs
log_info "Recent logs:"
echo ""
ssh root@${DROPLET_IP} "cd /opt/trendcortex && docker-compose logs --tail=20"

# Success message
echo ""
echo "=========================================="
log_info "Deployment complete!"
echo "=========================================="
echo ""
echo "Droplet IP: ${DROPLET_IP}"
echo ""
echo "Useful commands:"
echo ""
echo "  # SSH into droplet:"
echo "  ssh root@${DROPLET_IP}"
echo ""
echo "  # View logs:"
echo "  ssh root@${DROPLET_IP} 'cd /opt/trendcortex && docker-compose logs -f'"
echo ""
echo "  # Restart bot:"
echo "  ssh root@${DROPLET_IP} 'cd /opt/trendcortex && docker-compose restart'"
echo ""
echo "  # Stop bot:"
echo "  ssh root@${DROPLET_IP} 'cd /opt/trendcortex && docker-compose down'"
echo ""
echo "  # Update and redeploy:"
echo "  ./deploy-digital-ocean.sh"
echo ""
echo "=========================================="
