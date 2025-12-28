"""
Backtesting Configuration

This module contains all configuration settings for the backtesting environment,
including approved trading pairs, timeframes, date ranges, and file paths.
"""

from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# APPROVED TRADING PAIRS (Binance Spot Only)
# =============================================================================
# These are the ONLY pairs approved for backtesting and live trading
APPROVED_SYMBOLS = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "SOLUSDT",   # Solana
    "DOGEUSDT",  # Dogecoin
    "XRPUSDT",   # Ripple
    "ADAUSDT",   # Cardano
    "BNBUSDT",   # Binance Coin
    "LTCUSDT",   # Litecoin
]

# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================
# Supported Binance intervals
AVAILABLE_INTERVALS = [
    "1m",   # 1 minute
    "3m",   # 3 minutes
    "5m",   # 5 minutes
    "15m",  # 15 minutes
    "30m",  # 30 minutes
    "1h",   # 1 hour
    "2h",   # 2 hours
    "4h",   # 4 hours
    "6h",   # 6 hours
    "8h",   # 8 hours
    "12h",  # 12 hours
    "1d",   # 1 day
    "3d",   # 3 days
    "1w",   # 1 week
    "1M",   # 1 month
]

# Default intervals for backtesting
DEFAULT_INTERVALS = ["15m", "1h", "4h"]

# =============================================================================
# DATE RANGE CONFIGURATION
# =============================================================================
# Default backtest period: last 90 days
DEFAULT_DAYS_BACK = 90

# Calculate default start and end dates
DEFAULT_END_DATE = datetime.now()
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=DEFAULT_DAYS_BACK)

# Custom date ranges for different test scenarios
DATE_RANGES = {
    "last_week": {
        "start": datetime.now() - timedelta(days=7),
        "end": datetime.now()
    },
    "last_month": {
        "start": datetime.now() - timedelta(days=30),
        "end": datetime.now()
    },
    "last_quarter": {
        "start": datetime.now() - timedelta(days=90),
        "end": datetime.now()
    },
    "last_year": {
        "start": datetime.now() - timedelta(days=365),
        "end": datetime.now()
    },
    "2024_full": {
        "start": datetime(2024, 1, 1),
        "end": datetime(2024, 12, 31)
    },
    "2024_q4": {
        "start": datetime(2024, 10, 1),
        "end": datetime(2024, 12, 31)
    },
}

# =============================================================================
# BINANCE API CONFIGURATION
# =============================================================================
# Public API endpoint (no authentication required)
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = f"{BINANCE_BASE_URL}/api/v3/klines"

# API rate limits (to avoid hitting Binance limits)
# Binance allows 1200 requests per minute for spot API
MAX_REQUESTS_PER_MINUTE = 1000  # Stay below limit
REQUEST_DELAY_SECONDS = 0.1  # Small delay between requests

# Maximum candles per request (Binance limit)
MAX_CANDLES_PER_REQUEST = 1000

# =============================================================================
# FILE PATH CONFIGURATION
# =============================================================================
# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data file naming convention
# Format: {symbol}_{interval}_{start_date}_{end_date}.csv
DATA_FILE_PATTERN = "{symbol}_{interval}_{start}_{end}.csv"

# Results file naming convention
RESULTS_FILE_PATTERN = "backtest_{symbol}_{interval}_{timestamp}.csv"

# =============================================================================
# BACKTESTING CONFIGURATION
# =============================================================================
# Initial capital for backtesting (in USDT)
INITIAL_CAPITAL = 10000.0

# Position sizing (percentage of capital per trade)
POSITION_SIZE_PERCENT = 0.02  # 2% risk per trade

# Transaction fees (Binance spot trading fees)
# 0.1% maker/taker fee (without BNB discount)
MAKER_FEE = 0.001  # 0.1%
TAKER_FEE = 0.001  # 0.1%

# Slippage assumptions (in percentage)
SLIPPAGE = 0.0005  # 0.05% slippage

# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================
# Default EMA crossover strategy parameters
DEFAULT_STRATEGY_PARAMS = {
    "fast_ema": 20,       # Fast EMA period
    "slow_ema": 50,       # Slow EMA period
    "atr_period": 14,     # ATR period for volatility filter
    "atr_multiplier": 1.5,  # ATR multiplier for stop loss
    "min_atr": 0.001,     # Minimum ATR threshold (filter low volatility)
}

# =============================================================================
# INDICATOR CONFIGURATION
# =============================================================================
# Standard periods for technical indicators
INDICATOR_PERIODS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "ema_long": 200,
    "rsi": 14,
    "atr": 14,
    "bb_period": 20,
    "bb_std": 2,
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Log file format
LOG_FILE_FORMAT = "backtest_{date}.log"

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# Console output format
CONSOLE_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
FILE_LOG_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol is in the approved list.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
    
    Returns:
        True if symbol is approved, False otherwise
    """
    return symbol.upper() in APPROVED_SYMBOLS


def validate_interval(interval: str) -> bool:
    """
    Validate if an interval is supported.
    
    Args:
        interval: Timeframe interval (e.g., "1h")
    
    Returns:
        True if interval is valid, False otherwise
    """
    return interval in AVAILABLE_INTERVALS


def get_data_path(symbol: str, interval: str, start_date: datetime, end_date: datetime) -> Path:
    """
    Get the file path for historical data.
    
    Args:
        symbol: Trading pair symbol
        interval: Timeframe interval
        start_date: Start date
        end_date: End date
    
    Returns:
        Path object for the data file
    """
    filename = DATA_FILE_PATTERN.format(
        symbol=symbol,
        interval=interval,
        start=start_date.strftime("%Y%m%d"),
        end=end_date.strftime("%Y%m%d")
    )
    return DATA_DIR / filename


def get_results_path(symbol: str, interval: str) -> Path:
    """
    Get the file path for backtest results.
    
    Args:
        symbol: Trading pair symbol
        interval: Timeframe interval
    
    Returns:
        Path object for the results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_FILE_PATTERN.format(
        symbol=symbol,
        interval=interval,
        timestamp=timestamp
    )
    return RESULTS_DIR / filename


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def interval_to_milliseconds(interval: str) -> int:
    """
    Convert interval string to milliseconds.
    
    Args:
        interval: Interval string (e.g., "1h", "15m")
    
    Returns:
        Number of milliseconds in the interval
    """
    unit = interval[-1]
    value = int(interval[:-1])
    
    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    elif unit == 'w':
        return value * 7 * 24 * 60 * 60 * 1000
    elif unit == 'M':
        return value * 30 * 24 * 60 * 60 * 1000  # Approximate
    else:
        raise ValueError(f"Invalid interval unit: {unit}")


def print_config():
    """Print current configuration for debugging."""
    print("=" * 80)
    print("BACKTESTING CONFIGURATION")
    print("=" * 80)
    print(f"Approved Symbols: {', '.join(APPROVED_SYMBOLS)}")
    print(f"Default Intervals: {', '.join(DEFAULT_INTERVALS)}")
    print(f"Default Date Range: {DEFAULT_START_DATE.date()} to {DEFAULT_END_DATE.date()}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size: {POSITION_SIZE_PERCENT * 100}%")
    print(f"Maker/Taker Fee: {MAKER_FEE * 100}%")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    print_config()
    
    # Test validation
    print("\nValidation Tests:")
    print(f"BTCUSDT valid: {validate_symbol('BTCUSDT')}")
    print(f"INVALID valid: {validate_symbol('INVALID')}")
    print(f"1h valid: {validate_interval('1h')}")
    print(f"5s valid: {validate_interval('5s')}")
    
    # Test path generation
    print("\nPath Generation Tests:")
    test_path = get_data_path("BTCUSDT", "1h", DEFAULT_START_DATE, DEFAULT_END_DATE)
    print(f"Data path: {test_path}")
    
    test_results = get_results_path("BTCUSDT", "1h")
    print(f"Results path: {test_results}")
    
    # Test interval conversion
    print("\nInterval Conversion Tests:")
    print(f"1m = {interval_to_milliseconds('1m')} ms")
    print(f"1h = {interval_to_milliseconds('1h')} ms")
    print(f"1d = {interval_to_milliseconds('1d')} ms")
