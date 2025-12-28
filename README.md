# TrendCortex: Hybrid AI-Assisted Crypto Trading System

**A sophisticated Python-based cryptocurrency trading system designed for the WEEX AI Wars hackathon, featuring hybrid AI decision-making, advanced technical analysis, and robust risk management.**

## 🎯 Overview

TrendCortex combines rule-based technical analysis with machine learning and LLM-powered decision gates to execute intelligent, risk-managed trades on cryptocurrency futures markets via the WEEX API.

### Key Features

- **Async Architecture**: Built with Python 3.10+ async/await for high-performance concurrent operations
- **WEEX API Integration**: Full REST and WebSocket support with authentication
- **AI Logging**: Automatic logging of AI decisions to WEEX competition endpoint
- **Technical Analysis**: EMA, RSI, ATR, Bollinger Bands, and custom indicators
- **Hybrid AI Decision Engine**: ML model evaluation + LLM decision gate
- **Risk Management**: Multi-layer risk controls including leverage limits, position sizing, volatility filters
- **Structured Logging**: JSON-formatted logs for signals, decisions, and executions
- **Modular Design**: Clean separation of concerns for easy testing and extension

## 📁 Project Structure

```
TrendCortex/
├── trendcortex/                 # Main package directory
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── api_client.py            # WEEX API client (REST + WebSocket)
│   ├── data_manager.py          # Market data fetching and caching
│   ├── indicators.py            # Technical indicators (EMA, RSI, ATR, BB)
│   ├── signal_engine.py         # Rule-based signal generation
│   ├── model_integration.py     # ML/LLM decision framework
│   ├── risk_controller.py       # Risk management and validation
│   ├── execution.py             # Order execution engine
│   ├── ai_logger.py             # WEEX AI logging integration
│   ├── logger.py                # Structured logging system
│   └── utils.py                 # Utility functions
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_signal_engine.py
│   ├── test_risk_controller.py
│   └── test_indicators.py
├── docs/                        # Documentation
│   └── AI_LOGGING.md            # AI logging guide
├── logs/                        # Log output directory
│   ├── signals/
│   ├── decisions/
│   └── executions/
├── data/                        # Historical data cache
├── models/                      # Trained ML models
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── config.example.json          # Example configuration
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip or conda for package management
- WEEX API credentials (API Key, Secret Key, Passphrase)

### Installation

1. **Clone the repository**
   ```bash
   cd /home/rodrigodog/TrendCortex
   ```

2. **Create a virtual environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API credentials**
   ```bash
   cp config.example.json config.json
   ```
   
   Edit `config.json` and add your WEEX API credentials:
   ```json
   {
     "api": {
       "key": "YOUR_API_KEY",
       "secret": "YOUR_SECRET_KEY",
       "passphrase": "YOUR_PASSPHRASE"
     }
   }
   ```

### Configuration

The `config.json` file contains all system settings:

- **API Settings**: Credentials, base URL, rate limits
- **Trading Parameters**: Symbols, timeframes, leverage
- **Risk Management**: Max position size, stop loss, cooldown periods
- **Technical Indicators**: Periods, thresholds for signals
- **ML/LLM Settings**: Model paths, confidence thresholds

### Running the Bot

#### Dry-Run Mode (Recommended for Testing)

```bash
python main.py --dry-run
```

This mode will:
- Fetch live market data
- Generate signals
- Run AI decision logic
- Log all actions
- **NOT execute real trades**

#### Live Trading Mode

⚠️ **WARNING**: Only use after thorough testing in dry-run mode!

```bash
python main.py --live
```

#### Backtest Mode

```bash
python main.py --backtest --start-date 2025-01-01 --end-date 2025-01-31
```

### Monitoring

Logs are written to the `logs/` directory:

- `logs/signals/signals_YYYYMMDD.json` - Technical signals
- `logs/decisions/decisions_YYYYMMDD.json` - AI decisions
- `logs/executions/executions_YYYYMMDD.json` - Trade executions

View live logs:
```bash
tail -f logs/signals/signals_$(date +%Y%m%d).json | jq '.'
```

## 🧠 AI Model Integration

TrendCortex uses a hybrid AI approach:

### 1. Machine Learning Model

The `model_integration.py` module provides an abstraction layer for ML models:

```python
from trendcortex.model_integration import MLModelEvaluator

# Initialize model
evaluator = MLModelEvaluator(model_path="models/trained_model.pkl")

# Evaluate trade
result = await evaluator.evaluate_trade(
    symbol="cmt_btcusdt",
    signal=signal_data,
    market_data=market_data
)
```

**TODO**: Implement your ML model training pipeline:
- Feature engineering from market data
- Model training (e.g., XGBoost, Neural Network)
- Model serialization and versioning

### 2. LLM Decision Gate

The LLM gate provides human-like reasoning for complex market conditions:

```python
from trendcortex.model_integration import LLMDecisionGate

# Initialize LLM
llm = LLMDecisionGate(api_key="your_openai_key")

# Get decision
decision = await llm.make_decision(
    signal=signal_data,
    ml_score=ml_result,
    market_context=market_data
)
```

**TODO**: Integrate your preferred LLM provider:
- OpenAI GPT-4
- Anthropic Claude
- Local LLaMA models
- Custom fine-tuned models

### WEEX AI Logging

TrendCortex automatically logs all AI decisions to the WEEX competition endpoint:

```python
from trendcortex.ai_logger import create_decision_log, serialize_for_ai_log

# Logs are automatically created and uploaded
# See docs/AI_LOGGING.md for details
```

**Features**:
- ✅ Automatic logging of Strategy Generation, Decision Making, and Execution
- ✅ HMAC-SHA256 signed requests
- ✅ Links logs to actual order IDs
- ✅ Handles serialization of complex data types
- ✅ Error handling and retry logic

**Documentation**: See [`docs/AI_LOGGING.md`](docs/AI_LOGGING.md) for complete guide.

### Training Your Model

See `scripts/train_model.py` for a starter training pipeline:

```bash
python scripts/train_model.py --data data/historical/ --output models/
```

## 🔒 Risk Management

TrendCortex implements multiple risk control layers:

1. **Pre-Trade Validation**
   - Maximum leverage check (20x for WEEX competition)
   - Position size limits
   - Account balance verification
   - Volatility filters

2. **Active Trade Management**
   - Stop-loss enforcement
   - Take-profit targets
   - Trailing stops
   - Position cooldown periods

3. **Portfolio Limits**
   - Maximum open positions
   - Correlation limits
   - Exposure limits per asset

Configure risk parameters in `config.json`:

```json
{
  "risk": {
    "max_leverage": 20,
    "max_position_size_usdt": 1000,
    "max_portfolio_risk_percent": 5.0,
    "stop_loss_percent": 2.0,
    "take_profit_percent": 5.0,
    "cooldown_seconds": 300
  }
}
```

## 📊 Supported Trading Pairs

For WEEX AI Wars competition:
- cmt_btcusdt
- cmt_ethusdt
- cmt_solusdt
- cmt_dogeusdt
- cmt_xrpusdt
- cmt_adausdt
- cmt_bnbusdt
- cmt_ltcusdt

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_signal_engine.py

# With coverage
pytest --cov=trendcortex tests/
```

## 📈 Performance Metrics

TrendCortex tracks comprehensive performance metrics:

- Total PnL (realized + unrealized)
- Win rate
- Average profit per trade
- Sharpe ratio
- Maximum drawdown
- Trade frequency
- Signal accuracy

View metrics dashboard:
```bash
python scripts/analyze_performance.py --logs logs/executions/
```

## 🔧 Development

### Code Style

This project follows PEP 8 and uses type hints throughout:

```bash
# Format code
black trendcortex/

# Type checking
mypy trendcortex/

# Linting
pylint trendcortex/
```

### Adding New Indicators

1. Add indicator function to `trendcortex/indicators.py`
2. Update `trendcortex/signal_engine.py` to use new indicator
3. Add tests to `tests/test_indicators.py`
4. Update configuration schema

### Adding New Signals

1. Define signal logic in `trendcortex/signal_engine.py`
2. Configure parameters in `config.json`
3. Add tests to `tests/test_signal_engine.py`

## 🐛 Troubleshooting

### API Connection Issues

```bash
# Test connectivity
curl -s --max-time 10 "https://api-contract.weex.com/capi/v2/market/time"
```

### Common Issues

1. **521 Web Server is Down**
   - Your IP is not whitelisted
   - Contact WEEX support to whitelist your IP

2. **403 Forbidden**
   - Check API credentials in config.json
   - Verify passphrase is correct

3. **429 Too Many Requests**
   - Rate limit exceeded
   - Adjust `rate_limit_delay` in config.json

4. **Signature Verification Failed**
   - Ensure system time is synchronized (NTP)
   - Check that timestamp is within 30 seconds

## 📚 Resources

- [WEEX API Documentation](https://www.weex.com/api-doc/ai/intro)
- [Competition Rules](https://www.weex.com/events/ai-trading)
- [Technical Support](https://t.me/weex_support)

## 🏆 Competition Guidelines

### Important Rules

1. **Maximum Leverage**: 20x (platform allows 400x but competition limits to 20x)
2. **Initial Funds**: 1,000 USDT (reset before competition)
3. **Allowed Pairs**: Only the 8 specified cmt_ pairs
4. **Testing Phase**: Until January 5, 2026
5. **Competition Period**: Late February 2026 (17 days)

### Strategy Tips

- Start conservative during testing phase
- Monitor volatility and adjust leverage accordingly
- Use the LLM gate for high-uncertainty scenarios
- Log everything for post-analysis
- Test failure scenarios thoroughly

## 📄 License

This project is provided as-is for the WEEX AI Wars hackathon.

## 🤝 Contributing

This is a hackathon project. Feel free to fork and modify for your team's strategy!

## ⚠️ Disclaimer

Cryptocurrency trading carries significant risk. This software is provided for educational and competition purposes only. Trade at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

For technical issues or questions:
- Open an issue on GitHub
- Contact via WEEX competition TG group
- Email: support@trendcortex.ai (if applicable)

---

**Built for WEEX AI Wars: Alpha Awakens 🚀**

*Good luck and may the best algorithms win!*
