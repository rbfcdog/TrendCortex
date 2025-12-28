#!/bin/bash

# TrendCortex Setup Script
# 
# This script sets up the development environment for TrendCortex

set -e  # Exit on error

echo "====================================="
echo "TrendCortex Setup"
echo "====================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.10 or higher is required"
    echo "   Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs/signals
mkdir -p logs/decisions
mkdir -p logs/executions
mkdir -p data/cache
mkdir -p data/historical
mkdir -p models
echo "✅ Directories created"
echo ""

# Copy example config if config doesn't exist
if [ ! -f "config.json" ]; then
    echo "Creating config.json from example..."
    cp config.example.json config.json
    echo "✅ config.json created"
    echo ""
    echo "⚠️  IMPORTANT: Edit config.json and add your WEEX API credentials!"
    echo ""
else
    echo "⚠️  config.json already exists, skipping..."
    echo ""
fi

# Test import
echo "Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
from trendcortex import Config
from trendcortex.logger import setup_logging
print('✅ Core modules importable')
"
echo ""

# Display next steps
echo "====================================="
echo "Setup Complete! 🎉"
echo "====================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure API credentials:"
echo "   edit config.json"
echo ""
echo "2. Test API connection:"
echo "   python3 -c 'import asyncio; from trendcortex.api_client import WEEXAPIClient; from trendcortex.config import Config; asyncio.run(WEEXAPIClient(Config.load()).get_server_time())'"
echo ""
echo "3. Run in dry-run mode:"
echo "   python3 main.py --dry-run"
echo ""
echo "4. Run tests:"
echo "   pytest tests/ -v"
echo ""
echo "5. When ready for live trading:"
echo "   python3 main.py --live"
echo ""
echo "====================================="
echo "Documentation: README.md"
echo "====================================="
