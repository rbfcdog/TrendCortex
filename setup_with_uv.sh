#!/usr/bin/env bash
# TrendCortex - Complete Setup Script with UV Package Manager
# This script sets up the entire TrendCortex project including both the main trading bot and backtesting environment

set -e  # Exit on error

echo "========================================="
echo "🚀 TrendCortex Complete Setup with UV"
echo "========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV package manager not found!"
    echo "Please install UV first: https://github.com/astral-sh/uv"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ UV package manager found"
echo ""

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "📁 Project directory: $PROJECT_DIR"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment with UV..."
    uv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate
echo ""

# Install main project dependencies
echo "📦 Installing main TrendCortex dependencies..."
echo "   - Core trading libraries..."
uv pip install ccxt pandas numpy ta python-dotenv websocket-client
echo ""

# Install backtesting dependencies
echo "📦 Installing backtesting dependencies..."
cd backtesting
uv pip install pandas numpy requests
cd ..
echo ""

# Install optional dev dependencies
echo "📦 Installing optional development dependencies..."
uv pip install pytest pytest-cov black flake8
echo ""

echo "========================================="
echo "✅ Installation Complete!"
echo "========================================="
echo ""
echo "📊 Project Structure:"
echo "  ├── trendcortex/          # Main trading bot"
echo "  ├── backtesting/          # Backtesting environment"
echo "  ├── tests/                # Unit tests"
echo "  └── .venv/                # Virtual environment"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Test the backtesting system:"
echo "   cd backtesting"
echo "   python test_setup.py"
echo "   python quick_demo.py"
echo ""
echo "3. Run a backtest:"
echo "   python run_backtest.py --symbols BTCUSDT --interval 1h --days 7"
echo ""
echo "4. Configure your trading bot:"
echo "   cp .env.example .env  # Create config file"
echo "   nano .env             # Edit with your API keys"
echo ""
echo "5. Run the trading bot:"
echo "   python main.py"
echo ""
echo "📖 Documentation:"
echo "   - Main README: cat README.md"
echo "   - Backtesting README: cat backtesting/README.md"
echo ""
echo "========================================="
