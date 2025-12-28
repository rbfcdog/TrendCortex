# TrendCortex Project Structure

## 📁 Complete File Tree

```
TrendCortex/
├── main.py                          # Application entry point
├── test_api.py                      # API connection test script
├── setup.sh                         # Setup automation script
├── requirements.txt                 # Python dependencies
├── config.example.json              # Example configuration
├── config.json                      # Your configuration (git-ignored)
├── .gitignore                       # Git ignore rules
├── README.md                        # Full documentation
├── QUICKSTART.md                    # Quick start guide
│
├── trendcortex/                     # Main package
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Configuration management
│   ├── logger.py                    # Structured logging system
│   ├── utils.py                     # Utility functions
│   ├── api_client.py                # WEEX API client (REST + WebSocket)
│   ├── data_manager.py              # Market data fetching & caching
│   ├── indicators.py                # Technical indicators (EMA, RSI, ATR, etc.)
│   ├── signal_engine.py             # Trading signal generation
│   ├── model_integration.py         # ML/LLM decision framework
│   ├── risk_controller.py           # Risk management & validation
│   └── execution.py                 # Order execution engine
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_signal_engine.py        # Signal generation tests
│   ├── test_risk_controller.py      # Risk management tests
│   └── test_indicators.py           # Technical indicator tests
│
├── logs/                            # Log files (auto-created)
│   ├── signals/                     # Signal logs
│   ├── decisions/                   # AI decision logs
│   └── executions/                  # Execution logs
│
├── data/                            # Data storage (auto-created)
│   ├── cache/                       # Cached market data
│   └── historical/                  # Historical data for backtesting
│
└── models/                          # ML models (auto-created)
    └── trained_model.pkl            # Your trained model (placeholder)
```

## 🔧 Module Overview

### Core Trading System

#### `config.py` - Configuration Management
- Loads settings from JSON and environment variables
- Validates competition rules (20x leverage, allowed symbols)
- Type-safe configuration with Pydantic models
- **Key Features**: API credentials, risk limits, indicator parameters

#### `logger.py` - Structured Logging
- JSON-formatted logs for signals, decisions, executions
- Separate log streams for different components
- Automatic log rotation
- **Output**: `logs/signals/`, `logs/decisions/`, `logs/executions/`

#### `utils.py` - Utility Functions
- Time conversion (timestamps ↔ datetime)
- Price/size formatting and rounding
- Position sizing calculations
- PnL calculations
- **230+ lines** of helper functions

### Market Data Layer

#### `api_client.py` - WEEX API Client
- Async REST API calls with HMAC-SHA256 authentication
- Rate limiting and retry logic
- Public endpoints (ticker, orderbook, candles)
- Private endpoints (balance, positions, orders)
- **WebSocket support** (TODO: implementation stub)

#### `data_manager.py` - Data Management
- Fetches and caches market data
- Transforms API responses to pandas DataFrames
- Multi-timeframe data fetching
- Contract specifications caching
- **Cache TTL**: Configurable per data type

### Signal Generation

#### `indicators.py` - Technical Indicators
- **EMA**: Exponential Moving Average
- **RSI**: Relative Strength Index
- **ATR**: Average True Range
- **Bollinger Bands**: Volatility bands
- **MACD**: Moving Average Convergence Divergence
- **Stochastic**: Stochastic oscillator
- **OBV**: On-Balance Volume
- **ADX**: Average Directional Index
- Plus: Support/resistance detection, divergence detection

#### `signal_engine.py` - Signal Generation
- Rule-based signal strategies:
  - EMA crossover
  - RSI extreme levels
  - Bollinger Band breakouts
  - MACD crossovers
- Multi-indicator confirmation
- Confidence scoring (0.0-1.0)
- Stop loss and take profit calculation

### AI Decision Layer

#### `model_integration.py` - ML/LLM Integration
- **MLModelEvaluator**: Machine learning model wrapper
  - Feature extraction from market data
  - Model training interface (placeholder)
  - Trade evaluation with confidence scoring
- **LLMDecisionGate**: Large language model integration
  - Prompt building with market context
  - Decision parsing and validation
  - Provider-agnostic design (OpenAI, Anthropic, etc.)
- **HybridDecisionEngine**: Combines ML + LLM
  - Sequential evaluation pipeline
  - LLM veto capability
  - Confidence aggregation

### Risk Management

#### `risk_controller.py` - Risk Controller
- **Pre-trade validation**:
  - Minimum balance check
  - Maximum open positions
  - Position size limits
  - Leverage limits (enforces 20x max)
  - Volatility filter
  - Cooldown periods
  - Daily loss limits
- **Position sizing**:
  - Risk-based sizing
  - Stop loss distance consideration
  - Account balance percentage
- **Portfolio risk**:
  - Aggregate exposure calculation
  - Correlation limits (configurable)

### Order Execution

#### `execution.py` - Trade Executor
- **Order placement**:
  - Limit and market orders
  - Price rounding to tick size
  - Size rounding to increment
- **Order management**:
  - Fill confirmation polling
  - Partial fill handling
  - Order cancellation
- **Dry-run mode**:
  - Simulated execution
  - No real orders placed
  - Full logging maintained

### Application Entry

#### `main.py` - Main Trading Loop
- Component initialization
- Async trading loop orchestration
- Signal processing pipeline
- Error handling and recovery
- Graceful shutdown
- Performance tracking

## 📊 Data Flow

```
1. API Client → Raw market data
2. Data Manager → Processed DataFrames
3. Signal Engine → Trading signals
4. ML/LLM Models → Decision evaluation
5. Risk Controller → Validation & sizing
6. Executor → Order placement
7. Logger → Structured logs
```

## 🔑 Key Design Patterns

### Async/Await Architecture
All I/O operations are asynchronous for maximum performance:
```python
async with WEEXAPIClient(config) as client:
    data = await client.get_candles(symbol)
```

### Type Hints Everywhere
Complete type annotations for IDE support and type checking:
```python
def calculate_position_size(
    account_balance: float,
    risk_percent: float,
) -> float:
```

### Dataclass Models
Structured data with validation:
```python
@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    confidence: float
```

### Dependency Injection
Components receive dependencies via constructor:
```python
def __init__(self, api_client: WEEXAPIClient, config: Config):
```

## 🎯 Competition-Specific Features

### WEEX API Integration
- Full REST API implementation
- Authentication with HMAC-SHA256
- Error handling for all WEEX error codes
- Rate limiting compliance

### Rule Enforcement
- Maximum leverage: 20x (hard-coded check)
- Allowed symbols: Only 8 CMT pairs
- Position limits: Configurable
- Risk limits: Per-trade and portfolio-wide

### Logging & Monitoring
- All signals logged with indicators
- All decisions logged with reasoning
- All executions logged with results
- JSON format for easy parsing

## 🧩 Extension Points

### Add Your ML Model
Implement in `model_integration.py`:
```python
def train(self, data, labels):
    # Your model training here
    pass

async def evaluate_trade(self, signal, data):
    # Your model inference here
    pass
```

### Add Your LLM
Implement in `model_integration.py`:
```python
async def make_decision(self, signal, ml_decision, context):
    # Your LLM API call here
    pass
```

### Add Custom Indicators
Implement in `indicators.py`:
```python
def calculate_custom_indicator(data: pd.Series) -> pd.Series:
    # Your indicator logic here
    pass
```

### Add Custom Signals
Implement in `signal_engine.py`:
```python
def _check_custom_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
    # Your signal logic here
    pass
```

## 📦 Dependencies Overview

### Core
- **aiohttp**: Async HTTP client
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Technical Analysis
- **pandas-ta**: Technical indicators
- **ta-lib**: Advanced indicators (optional)

### ML/AI (Optional)
- **scikit-learn**: Traditional ML
- **xgboost**: Gradient boosting
- **openai**: GPT integration
- **anthropic**: Claude integration

### Utilities
- **pydantic**: Configuration validation
- **python-json-logger**: Structured logging
- **tenacity**: Retry logic

### Testing
- **pytest**: Testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting

## 🚀 Getting Started

1. **Setup**: `./setup.sh`
2. **Configure**: Edit `config.json`
3. **Test API**: `./test_api.py`
4. **Dry Run**: `python3 main.py --dry-run`
5. **Run Tests**: `pytest tests/ -v`
6. **Go Live**: `python3 main.py --live`

## 📖 Documentation Files

- **README.md**: Complete documentation (500+ lines)
- **QUICKSTART.md**: 5-minute quick start
- **PROJECT_STRUCTURE.md**: This file
- **config.example.json**: Configuration template

## 🎓 Learning Resources

- WEEX API: https://www.weex.com/api-doc/ai/intro
- Competition: https://www.weex.com/events/ai-trading
- Python Async: https://docs.python.org/3/library/asyncio.html
- Pandas: https://pandas.pydata.org/docs/
- Technical Analysis: https://www.investopedia.com/

## 📊 Project Statistics

- **Total Files**: 20+
- **Python Modules**: 12
- **Lines of Code**: 3,500+
- **Test Files**: 3
- **Documentation**: 1,000+ lines
- **Configuration Options**: 100+

## ✅ Production Ready Features

- ✅ Async architecture
- ✅ Type hints throughout
- ✅ Structured logging
- ✅ Error handling
- ✅ Rate limiting
- ✅ Retry logic
- ✅ Dry-run mode
- ✅ Unit tests
- ✅ Configuration validation
- ✅ Graceful shutdown
- ✅ Performance tracking

## 🔜 TODO Items (Marked in Code)

Search for `TODO:` in codebase to find:
- WebSocket implementation
- ML model training pipeline
- LLM API integration
- Backtest engine
- Performance analytics
- Additional signal strategies
- More unit tests

## 🤝 Team Collaboration

### Git Workflow
```bash
# Clone repository
git clone <your-repo>

# Create feature branch
git checkout -b feature/your-feature

# Commit changes
git add .
git commit -m "Add feature"

# Push to remote
git push origin feature/your-feature
```

### Code Style
- PEP 8 compliance
- Black formatting
- Type hints required
- Docstrings for all public functions

---

**Built for WEEX AI Wars: Alpha Awakens 🚀**

*Last updated: December 26, 2025*
