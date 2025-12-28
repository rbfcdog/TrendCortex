# TrendCortex Backtesting Environment

Complete Python backtesting framework for testing cryptocurrency trading strategies on historical data from Binance. **No API keys required** - uses public data endpoints only.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Approved Trading Pairs](#approved-trading-pairs)
- [Extending the System](#extending-the-system)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- ✅ **Historical Data Fetching**: Download unlimited OHLCV data from Binance (automatic batching)
- ✅ **Smart Caching**: CSV-based caching to avoid redundant API calls
- ✅ **Technical Indicators**: EMA, ATR, RSI, Bollinger Bands, MACD, and more
- ✅ **Strategy Framework**: Pluggable strategy system with default EMA crossover strategy
- ✅ **Realistic Simulation**: Models transaction fees, slippage, and position sizing
- ✅ **Performance Metrics**: Win rate, P&L, Sharpe ratio, max drawdown, profit factor
- ✅ **Multi-Symbol Support**: Backtest across multiple pairs simultaneously
- ✅ **Flexible Timeframes**: Supports 1m, 5m, 15m, 30m, 1h, 4h, 1d, and more
- ✅ **Detailed Logging**: Track every trade with entry/exit prices, P&L, and hold time
- ✅ **No API Keys Needed**: Uses Binance public endpoints only

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Python Dependencies

```bash
cd backtesting
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `requests` - HTTP requests for API calls

### Step 2: Verify Installation

```bash
python config.py
```

You should see the configuration printed successfully.

## 🎯 Quick Start

### 1. Fetch Historical Data

```bash
# Test data fetching for Bitcoin
python data_fetcher.py
```

This will:
- Fetch 7 days of hourly BTC data
- Save to `data/BTCUSDT_1h_YYYYMMDD_YYYYMMDD.csv`
- Demonstrate caching (second call is instant)

### 2. Run Your First Backtest

```bash
# Backtest Bitcoin on 1-hour timeframe for last 90 days
python run_backtest.py --symbols BTCUSDT --interval 1h --days 90
```

Output:
```
================================================================================
BACKTEST PERFORMANCE SUMMARY
================================================================================

Strategy: EMACrossoverStrategy
Initial Capital: $10,000.00
Final Capital: $10,523.50
Total Return: +5.24%
Total P&L: +$523.50

Trades: 15
Winning: 9 (60.0%)
Losing: 6 (40.0%)

Average Win: +$125.30
Average Loss: -$85.20
Profit Factor: 1.67

Max Drawdown: -8.50%
Sharpe Ratio: 1.32
================================================================================
```

### 3. Backtest Multiple Symbols

```bash
# Backtest BTC, ETH, and SOL on 15m timeframe
python run_backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 15m --days 30
```

### 4. Backtest All Approved Symbols

```bash
# Backtest all 8 approved pairs
python run_backtest.py --all-symbols --interval 1h --days 90 --save-results
```

## 📚 Usage Examples

### Example 1: Basic Backtest

```bash
python run_backtest.py --symbols BTCUSDT --interval 1h --days 90
```

### Example 2: Custom Date Range

```bash
python run_backtest.py --symbols ETHUSDT --interval 4h --date-range 2024-01-01 2024-12-31
```

### Example 3: Multiple Symbols with Custom Strategy Parameters

```bash
python run_backtest.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --interval 1h \
  --days 180 \
  --fast-ema 10 \
  --slow-ema 30 \
  --atr-period 14 \
  --save-results
```

### Example 4: Force Data Refresh

```bash
# Ignore cache and fetch fresh data
python run_backtest.py --symbols BTCUSDT --interval 1h --days 30 --no-cache
```

### Example 5: Verbose Output

```bash
# Enable debug logging for troubleshooting
python run_backtest.py --symbols BTCUSDT --interval 1h --days 90 --verbose
```

## 📁 Project Structure

```
backtesting/
├── config.py              # Configuration settings
├── data_fetcher.py        # Historical data fetching from Binance
├── indicators.py          # Technical indicator calculations
├── backtester.py          # Backtesting engine
├── run_backtest.py        # CLI runner script
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── data/                 # Cached historical data (CSV files)
│   ├── BTCUSDT_1h_20240101_20241231.csv
│   ├── ETHUSDT_1h_20240101_20241231.csv
│   └── ...
│
├── results/              # Backtest results
│   ├── backtest_BTCUSDT_1h_20241226_143022.csv
│   ├── equity_BTCUSDT_1h_20241226_143022.csv
│   └── ...
│
└── logs/                 # Log files (if file logging enabled)
```

### Module Descriptions

#### `config.py`
- Approved trading pairs list
- Timeframe configurations
- Default date ranges
- File paths and naming conventions
- API endpoints and rate limits
- Validation functions

#### `data_fetcher.py`
- `fetch_ohlcv()` - Single API call to Binance
- `fetch_ohlcv_batch()` - Automatic batching for large date ranges
- `get_historical_data()` - High-level function with caching
- `save_to_csv()` / `load_from_csv()` - Data persistence
- Time conversion utilities

#### `indicators.py`
- `compute_ema()` - Exponential Moving Average
- `compute_atr()` - Average True Range
- `compute_rsi()` - Relative Strength Index
- `compute_bollinger_bands()` - Bollinger Bands
- `compute_macd()` - MACD indicator
- `add_all_indicators()` - Convenience function

#### `backtester.py`
- `Strategy` - Base class for strategies
- `EMACrossoverStrategy` - Default EMA crossover strategy
- `Backtester` - Main backtesting engine
- Trade execution simulation
- P&L calculation
- Performance metrics

#### `run_backtest.py`
- Command-line interface
- Argument parsing
- Multi-symbol backtesting
- Results aggregation
- CSV export

## 🎯 Approved Trading Pairs

The following 8 Binance spot pairs are approved for backtesting:

| Symbol | Name | Notes |
|--------|------|-------|
| BTCUSDT | Bitcoin | Highest liquidity |
| ETHUSDT | Ethereum | Second largest |
| SOLUSDT | Solana | High volatility |
| DOGEUSDT | Dogecoin | Meme coin |
| XRPUSDT | Ripple | Banking focus |
| ADAUSDT | Cardano | PoS blockchain |
| BNBUSDT | Binance Coin | Exchange token |
| LTCUSDT | Litecoin | Bitcoin fork |

**Why only these pairs?**
- High liquidity (tight spreads)
- Available on Binance spot
- Reliable historical data
- Popular trading pairs
- Reduces complexity

## 🔧 Extending the System

### Creating Custom Strategies

Create a new strategy by inheriting from the `Strategy` base class:

```python
# custom_strategy.py
from backtester import Strategy, SignalType
import pandas as pd

class MyCustomStrategy(Strategy):
    """
    My custom trading strategy.
    """
    
    def __init__(self, params: dict = None):
        if params is None:
            params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
            }
        super().__init__(params)
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> SignalType:
        """
        Generate signal based on RSI.
        """
        if index < 1:
            return SignalType.HOLD
        
        current = df.iloc[index]
        rsi = current['rsi']
        
        # Buy when RSI crosses above oversold level
        if rsi > self.params['rsi_oversold']:
            if df.iloc[index-1]['rsi'] <= self.params['rsi_oversold']:
                return SignalType.BUY
        
        # Sell when RSI crosses below overbought level
        if rsi < self.params['rsi_overbought']:
            if df.iloc[index-1]['rsi'] >= self.params['rsi_overbought']:
                return SignalType.SELL
        
        return SignalType.HOLD
```

Then use it:

```python
from custom_strategy import MyCustomStrategy
from backtester import Backtester
from data_fetcher import get_historical_data

# Fetch data
df = get_historical_data("BTCUSDT", "1h", start_date, end_date)

# Create custom strategy
strategy = MyCustomStrategy()

# Run backtest
backtester = Backtester(df, strategy)
results = backtester.run()
backtester.print_summary()
```

### Adding Custom Indicators

Add indicators to `indicators.py`:

```python
def compute_my_indicator(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute my custom indicator.
    """
    # Your indicator calculation here
    return result
```

### Customizing Position Sizing

Modify in `backtester.py` `open_position()` method:

```python
# Fixed position size
position_value = 1000  # Always trade $1000

# Risk-based sizing
risk_amount = self.capital * 0.01  # Risk 1% per trade
position_value = risk_amount / (atr * atr_multiplier)

# Volatility-adjusted sizing
position_value = self.capital * (1 / atr)
```

## 📊 Performance Metrics

The backtester calculates the following metrics:

### Basic Metrics
- **Total Return**: Overall percentage return
- **Total P&L**: Net profit/loss in USDT
- **Total Trades**: Number of trades executed
- **Win Rate**: Percentage of winning trades
- **Average Win**: Average profit on winning trades
- **Average Loss**: Average loss on losing trades

### Advanced Metrics
- **Profit Factor**: (Total Win Amount) / (Total Loss Amount)
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Average Hold Time**: Average time per trade

### Example Output

```
================================================================================
BACKTEST PERFORMANCE SUMMARY
================================================================================

Strategy: EMACrossoverStrategy
Initial Capital: $10,000.00
Final Capital: $11,245.50
Total Return: +12.46%
Total P&L: +$1,245.50
Total Fees: $58.30

Trades: 24
Winning: 15 (62.5%)
Losing: 9 (37.5%)

Average Win: +$125.60
Average Loss: -$75.20
Profit Factor: 2.51

Max Drawdown: -7.85%
Sharpe Ratio: 1.67
================================================================================
```

## 🐛 Troubleshooting

### Issue: "No data returned from API"

**Cause**: Binance API rate limiting or invalid date range

**Solution**:
```bash
# Try smaller date range
python run_backtest.py --symbols BTCUSDT --interval 1h --days 7

# Add delay between requests (already built-in)
# Check config.py REQUEST_DELAY_SECONDS
```

### Issue: "Import errors" (pandas, numpy, requests)

**Cause**: Dependencies not installed

**Solution**:
```bash
pip install -r requirements.txt

# If still failing, try:
pip install pandas numpy requests --upgrade
```

### Issue: "Permission denied" when saving files

**Cause**: No write permission in directory

**Solution**:
```bash
# Check permissions
ls -la

# Create directories manually
mkdir -p data results logs
chmod 755 data results logs
```

### Issue: Backtest runs but no trades executed

**Cause**: Strategy parameters too strict or insufficient data

**Solution**:
```bash
# Use more relaxed parameters
python run_backtest.py \
  --symbols BTCUSDT \
  --interval 1h \
  --days 180 \
  --fast-ema 10 \
  --slow-ema 20

# Or check if data has indicators
python -c "from data_fetcher import get_historical_data; \
           from indicators import add_all_indicators; \
           from datetime import datetime, timedelta; \
           df = get_historical_data('BTCUSDT', '1h', \
                datetime.now() - timedelta(days=30), datetime.now()); \
           df = add_all_indicators(df); \
           print(df[['close', 'ema_fast', 'ema_slow']].tail())"
```

### Issue: Slow performance with large datasets

**Solution**:
```bash
# Use shorter timeframes or date ranges
python run_backtest.py --symbols BTCUSDT --interval 4h --days 90

# Or limit symbols
python run_backtest.py --symbols BTCUSDT ETHUSDT --interval 1h --days 30
```

## 📈 Understanding the Data

### Binance OHLCV Format

Each candle contains:
- `open_time`: Candle open timestamp
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume (in base currency)
- `close_time`: Candle close timestamp
- `quote_volume`: Trading volume (in quote currency, e.g., USDT)
- `trades`: Number of trades in this candle

### Data Limits

- **Binance Limit**: 1000 candles per API call
- **Solution**: Automatic batching (handled by `fetch_ohlcv_batch()`)
- **Historical Limit**: Binance stores data back to 2017 for most pairs
- **Rate Limit**: 1200 requests/minute (we use 1000/minute to be safe)

## 🎓 Learning Resources

### Understanding EMA Crossover Strategy

The default strategy uses:
1. **Fast EMA (20)**: Reacts quickly to price changes
2. **Slow EMA (50)**: Smooths out noise
3. **Crossover**: When fast crosses above slow = BUY signal
4. **ATR Filter**: Only trade when volatility is above threshold

**Why this works**:
- Identifies trend changes early
- Filters out choppy markets (ATR)
- Simple and reliable

### Reading Backtest Results

Example trade:
```
OPENED LONG @ $45,230.00 | Qty: 0.022 | SL: $44,850.00 | TP: $46,120.00
CLOSED SIGNAL @ $45,890.00 | P&L: +$14.52 (+1.46%) | Hold: 8.5h
```

- **Entry**: Bought at $45,230
- **Quantity**: 0.022 BTC
- **Stop Loss**: Would exit if price drops to $44,850
- **Take Profit**: Would exit if price rises to $46,120
- **Exit**: Closed by strategy signal at $45,890
- **P&L**: Made $14.52 profit (1.46% return)
- **Hold Time**: Position held for 8.5 hours

## 🚨 Important Notes

### This is for Backtesting Only

- **DO NOT** use this for live trading without extensive testing
- Backtests use historical data - past performance ≠ future results
- Real trading involves additional risks (latency, order rejection, etc.)

### No API Keys Required

- This system uses **public Binance endpoints only**
- You don't need a Binance account
- No authentication or API key setup required
- Perfect for research and strategy development

### Data Disclaimer

- Data is fetched from Binance public API
- We assume data accuracy but can't guarantee it
- Always verify critical results with multiple sources

## 📞 Support

### Getting Help

1. **Check logs**: Look in `logs/` directory for error messages
2. **Verbose mode**: Run with `--verbose` flag for detailed output
3. **Test components**: Run individual modules to isolate issues:
   ```bash
   python data_fetcher.py  # Test data fetching
   python indicators.py     # Test indicator calculations
   python backtester.py     # Test backtest engine
   ```

### Common Commands

```bash
# Test everything is working
python config.py
python data_fetcher.py
python indicators.py
python backtester.py

# Quick backtest
python run_backtest.py --symbols BTCUSDT --interval 1h --days 7

# Full backtest with all features
python run_backtest.py \
  --all-symbols \
  --interval 1h \
  --days 180 \
  --save-results \
  --verbose
```

## 📝 Next Steps

1. **Run Examples**: Try the quick start examples above
2. **Experiment with Parameters**: Adjust EMA periods, ATR thresholds
3. **Create Custom Strategies**: Build your own trading logic
4. **Analyze Results**: Study which pairs and timeframes work best
5. **Optimize**: Fine-tune parameters using optimization techniques
6. **Paper Trade**: Test strategies in real-time (not included here)
7. **Live Trading**: Integrate with TrendCortex bot (separate system)

## 🎉 You're Ready!

Start by running:

```bash
python run_backtest.py --symbols BTCUSDT --interval 1h --days 90
```

Happy backtesting! 🚀
