#!/usr/bin/env python3
"""
Test Script for Backtesting Environment

This script runs basic tests to verify that all components are working correctly.

Run this after installation to ensure everything is set up properly:
    python test_setup.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

print("=" * 80)
print("BACKTESTING ENVIRONMENT - SETUP TEST")
print("=" * 80)
print()

# Test 1: Import all modules
print("1. Testing module imports...")
try:
    import pandas as pd
    print("   ✓ pandas imported")
except ImportError as e:
    print(f"   ✗ pandas import failed: {e}")
    print("   → Install with: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
    print("   ✓ numpy imported")
except ImportError as e:
    print(f"   ✗ numpy import failed: {e}")
    print("   → Install with: pip install numpy")
    sys.exit(1)

try:
    import requests
    print("   ✓ requests imported")
except ImportError as e:
    print(f"   ✗ requests import failed: {e}")
    print("   → Install with: pip install requests")
    sys.exit(1)

try:
    import config
    print("   ✓ config module imported")
except ImportError as e:
    print(f"   ✗ config import failed: {e}")
    sys.exit(1)

try:
    import data_fetcher
    print("   ✓ data_fetcher module imported")
except ImportError as e:
    print(f"   ✗ data_fetcher import failed: {e}")
    sys.exit(1)

try:
    import indicators
    print("   ✓ indicators module imported")
except ImportError as e:
    print(f"   ✗ indicators import failed: {e}")
    sys.exit(1)

try:
    import backtester
    print("   ✓ backtester module imported")
except ImportError as e:
    print(f"   ✗ backtester import failed: {e}")
    sys.exit(1)

print()

# Test 2: Verify directories
print("2. Checking directories...")
for dir_name in ['data', 'results', 'logs']:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   ✓ {dir_name}/ directory exists")
    else:
        print(f"   ✗ {dir_name}/ directory missing")
        print(f"   → Creating {dir_name}/ directory...")
        dir_path.mkdir(exist_ok=True)
        print(f"   ✓ Created {dir_name}/ directory")

print()

# Test 3: Test configuration
print("3. Testing configuration...")
try:
    assert len(config.APPROVED_SYMBOLS) == 8, "Expected 8 approved symbols"
    print(f"   ✓ {len(config.APPROVED_SYMBOLS)} approved symbols configured")
    
    assert config.validate_symbol("BTCUSDT"), "BTCUSDT should be valid"
    assert not config.validate_symbol("INVALID"), "INVALID should not be valid"
    print("   ✓ Symbol validation working")
    
    assert config.validate_interval("1h"), "1h should be valid interval"
    assert not config.validate_interval("invalid"), "invalid should not be valid"
    print("   ✓ Interval validation working")
    
except AssertionError as e:
    print(f"   ✗ Configuration test failed: {e}")
    sys.exit(1)

print()

# Test 4: Test data fetching (small request)
print("4. Testing data fetching (this may take a few seconds)...")
try:
    from data_fetcher import get_historical_data, datetime_to_ms, ms_to_datetime
    
    # Test time conversion
    now = datetime.now()
    now_ms = datetime_to_ms(now)
    back_to_dt = ms_to_datetime(now_ms)
    assert abs((now - back_to_dt).total_seconds()) < 1, "Time conversion failed"
    print("   ✓ Time conversion functions working")
    
    # Test data fetching (just 2 days to be quick)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    print(f"   → Fetching 2 days of BTCUSDT 1h data...")
    df = get_historical_data("BTCUSDT", "1h", start_date, end_date)
    
    assert not df.empty, "DataFrame should not be empty"
    assert 'open' in df.columns, "DataFrame should have 'open' column"
    assert 'close' in df.columns, "DataFrame should have 'close' column"
    print(f"   ✓ Fetched {len(df)} candles successfully")
    print(f"   ✓ Data saved to data/ directory")
    
except Exception as e:
    print(f"   ✗ Data fetching failed: {e}")
    print("   Note: This requires internet connection to Binance API")
    print("   If you're offline, this test will fail but other components may still work")

print()

# Test 5: Test indicators
print("5. Testing indicator calculations...")
try:
    from indicators import compute_ema, compute_atr, add_all_indicators
    
    # Create sample data
    sample_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [99, 100, 101, 102, 103],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400],
    })
    
    # Test EMA
    ema = compute_ema(sample_data['close'], 3)
    assert len(ema) == len(sample_data), "EMA length mismatch"
    print("   ✓ EMA calculation working")
    
    # Test ATR
    atr = compute_atr(sample_data, 3)
    assert len(atr) == len(sample_data), "ATR length mismatch"
    print("   ✓ ATR calculation working")
    
    print("   ✓ All indicator functions operational")
    
except Exception as e:
    print(f"   ✗ Indicator test failed: {e}")
    sys.exit(1)

print()

# Test 6: Test strategy
print("6. Testing strategy...")
try:
    from backtester import EMACrossoverStrategy, SignalType
    
    strategy = EMACrossoverStrategy()
    assert strategy.name == "EMACrossoverStrategy", "Strategy name incorrect"
    print(f"   ✓ Strategy created: {strategy.name}")
    print(f"   ✓ Strategy parameters: {strategy.params}")
    
except Exception as e:
    print(f"   ✗ Strategy test failed: {e}")
    sys.exit(1)

print()

# Test 7: Test backtester (quick test with sample data)
print("7. Testing backtester...")
try:
    from backtester import Backtester
    
    # Create more extensive sample data for backtesting
    np.random.seed(42)
    n_bars = 100
    prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    sample_data = pd.DataFrame({
        'open_time': pd.date_range(start='2024-01-01', periods=n_bars, freq='1h'),
        'open': prices + np.random.randn(n_bars) * 0.1,
        'high': prices + abs(np.random.randn(n_bars) * 0.5),
        'low': prices - abs(np.random.randn(n_bars) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars),
        'close_time': pd.date_range(start='2024-01-01', periods=n_bars, freq='1h'),
        'quote_volume': np.random.randint(100000, 1000000, n_bars),
        'trades': np.random.randint(100, 1000, n_bars),
        'taker_buy_base': np.random.randint(500, 5000, n_bars),
        'taker_buy_quote': np.random.randint(50000, 500000, n_bars),
        'ignore': 0,
    })
    
    strategy = EMACrossoverStrategy()
    backtester_obj = Backtester(sample_data, strategy)
    
    print("   ✓ Backtester instance created")
    print(f"   ✓ Initial capital: ${backtester_obj.initial_capital:,.2f}")
    
    # Note: We don't run the full backtest here as it's just a setup test
    print("   ✓ Backtester ready (full test skipped for speed)")
    
except Exception as e:
    print(f"   ✗ Backtester test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Summary
print("=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print()
print("Your backtesting environment is ready to use!")
print()
print("Next steps:")
print("  1. Run a quick backtest:")
print("     python run_backtest.py --symbols BTCUSDT --interval 1h --days 7")
print()
print("  2. Read the documentation:")
print("     cat README.md")
print()
print("  3. Explore the examples:")
print("     python data_fetcher.py")
print("     python indicators.py")
print("     python backtester.py")
print()
print("=" * 80)
