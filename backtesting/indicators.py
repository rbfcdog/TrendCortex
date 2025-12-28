"""
Technical Indicators Module

This module provides functions to compute technical indicators used in backtesting
strategies. All functions use vectorized pandas operations for performance.

Supported Indicators:
- EMA (Exponential Moving Average)
- ATR (Average True Range)
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
"""

import pandas as pd
import numpy as np
from typing import Tuple


# =============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# =============================================================================

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Exponential Moving Average.
    
    EMA gives more weight to recent prices, making it more responsive to
    price changes compared to Simple Moving Average (SMA).
    
    Formula:
        EMA = Price(t) * k + EMA(y) * (1 - k)
        where k = 2 / (period + 1)
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for EMA calculation
    
    Returns:
        Series with EMA values
    
    Example:
        >>> close_prices = df['close']
        >>> ema_20 = compute_ema(close_prices, 20)
        >>> df['ema_20'] = ema_20
    """
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Simple Moving Average.
    
    SMA is the average of prices over a specified period.
    
    Args:
        series: Price series
        period: Number of periods
    
    Returns:
        Series with SMA values
    
    Example:
        >>> sma_50 = compute_sma(df['close'], 50)
    """
    return series.rolling(window=period).mean()


# =============================================================================
# AVERAGE TRUE RANGE (ATR)
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    ATR measures market volatility by calculating the average of true ranges
    over a specified period. It's useful for:
    - Position sizing
    - Stop loss placement
    - Volatility filtering
    
    True Range is the maximum of:
    - Current High - Current Low
    - Abs(Current High - Previous Close)
    - Abs(Current Low - Previous Close)
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Number of periods for ATR calculation (default: 14)
    
    Returns:
        Series with ATR values
    
    Example:
        >>> atr = compute_atr(df, 14)
        >>> df['atr'] = atr
        >>> # Use ATR for stop loss: stop = entry_price - (2 * atr)
    """
    # Calculate True Range components
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    # True Range is the maximum of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR is the EMA of True Range
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr


# =============================================================================
# RELATIVE STRENGTH INDEX (RSI)
# =============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and magnitude of
    price changes. It oscillates between 0 and 100:
    - Above 70: Overbought
    - Below 30: Oversold
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for RSI calculation (default: 14)
    
    Returns:
        Series with RSI values (0-100)
    
    Example:
        >>> rsi = compute_rsi(df['close'], 14)
        >>> df['rsi'] = rsi
        >>> # Buy when RSI crosses above 30 (oversold)
        >>> # Sell when RSI crosses below 70 (overbought)
    """
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# =============================================================================
# BOLLINGER BANDS
# =============================================================================

def compute_bollinger_bands(
    series: pd.Series, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.
    
    Bollinger Bands consist of:
    - Middle Band: Simple Moving Average
    - Upper Band: SMA + (Standard Deviation * multiplier)
    - Lower Band: SMA - (Standard Deviation * multiplier)
    
    Bollinger Bands expand during volatile periods and contract during
    calm periods. They're useful for:
    - Identifying overbought/oversold conditions
    - Volatility measurement
    - Mean reversion strategies
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for SMA (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    
    Example:
        >>> upper, middle, lower = compute_bollinger_bands(df['close'], 20, 2)
        >>> df['bb_upper'] = upper
        >>> df['bb_middle'] = middle
        >>> df['bb_lower'] = lower
        >>> # Buy when price touches lower band
        >>> # Sell when price touches upper band
    """
    # Calculate middle band (SMA)
    middle_band = series.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = series.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


# =============================================================================
# MACD (Moving Average Convergence Divergence)
# =============================================================================

def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line
    
    Trading signals:
    - Bullish: MACD crosses above Signal
    - Bearish: MACD crosses below Signal
    
    Args:
        series: Price series (typically close prices)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    
    Example:
        >>> macd, signal, histogram = compute_macd(df['close'])
        >>> df['macd'] = macd
        >>> df['macd_signal'] = signal
        >>> df['macd_histogram'] = histogram
        >>> # Buy when MACD crosses above signal
        >>> # Sell when MACD crosses below signal
    """
    # Calculate fast and slow EMAs
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


# =============================================================================
# ADDITIONAL INDICATORS
# =============================================================================

def compute_stochastic(
    df: pd.DataFrame,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Stochastic Oscillator.
    
    The Stochastic Oscillator compares a closing price to its price range
    over a given time period. It oscillates between 0 and 100:
    - Above 80: Overbought
    - Below 20: Oversold
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default: 14)
        smooth_k: Smoothing for %K line (default: 3)
        smooth_d: Smoothing for %D line (default: 3)
    
    Returns:
        Tuple of (%K, %D)
    """
    # Calculate %K
    lowest_low = df['low'].rolling(window=period).min()
    highest_high = df['high'].rolling(window=period).max()
    
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    k_percent = k_percent.rolling(window=smooth_k).mean()
    
    # Calculate %D (signal line)
    d_percent = k_percent.rolling(window=smooth_d).mean()
    
    return k_percent, d_percent


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    Compute On-Balance Volume (OBV).
    
    OBV is a volume-based indicator that measures buying and selling pressure.
    - Rising OBV: Buying pressure
    - Falling OBV: Selling pressure
    
    Args:
        df: DataFrame with 'close' and 'volume' columns
    
    Returns:
        Series with OBV values
    """
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute Volume Weighted Average Price (VWAP).
    
    VWAP is the average price weighted by volume. It's used to determine
    the true average price of a security.
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
    
    Returns:
        Series with VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


# =============================================================================
# MULTI-INDICATOR HELPER
# =============================================================================

def add_all_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add all common technical indicators to a DataFrame.
    
    This is a convenience function that adds all the most commonly used
    indicators to your OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        config: Optional dict with indicator periods (uses defaults if None)
    
    Returns:
        DataFrame with added indicator columns
    
    Example:
        >>> df = get_historical_data("BTCUSDT", "1h", start, end)
        >>> df = add_all_indicators(df)
        >>> print(df.columns)
        # Now has EMA, ATR, RSI, Bollinger Bands, MACD, etc.
    """
    # Default configuration
    if config is None:
        config = {
            'ema_fast': 20,
            'ema_slow': 50,
            'ema_long': 200,
            'atr_period': 14,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
        }
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # EMAs - support both naming conventions
    fast_ema_period = config.get('ema_fast') or config.get('fast_ema', 20)
    slow_ema_period = config.get('ema_slow') or config.get('slow_ema', 50)
    long_ema_period = config.get('ema_long', 200)
    atr_period = config.get('atr_period', 14)
    rsi_period = config.get('rsi_period', 14)
    bb_period = config.get('bb_period', 20)
    bb_std = config.get('bb_std', 2.0)
    
    df['ema_fast'] = compute_ema(df['close'], fast_ema_period)
    df['ema_slow'] = compute_ema(df['close'], slow_ema_period)
    df['ema_long'] = compute_ema(df['close'], long_ema_period)
    
    # ATR
    df['atr'] = compute_atr(df, atr_period)
    
    # RSI
    df['rsi'] = compute_rsi(df['close'], rsi_period)
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = compute_bollinger_bands(
        df['close'], 
        bb_period, 
        bb_std
    )
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_histogram'] = compute_macd(df['close'])
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = compute_stochastic(df)
    
    # OBV
    df['obv'] = compute_obv(df)
    
    # VWAP
    df['vwap'] = compute_vwap(df)
    
    return df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of indicator functions.
    
    Run this file directly to see indicators in action:
        python indicators.py
    """
    from datetime import datetime, timedelta
    from data_fetcher import get_historical_data
    
    print("=" * 80)
    print("Technical Indicators Example")
    print("=" * 80)
    
    # Fetch some data
    print("\n1. Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = get_historical_data("BTCUSDT", "1h", start_date, end_date)
    print(f"   Loaded {len(df)} candles")
    
    # Add all indicators
    print("\n2. Computing all indicators...")
    df = add_all_indicators(df)
    print(f"   Added {len(df.columns) - 12} indicator columns")
    
    # Display sample
    print("\n3. Sample data with indicators:")
    print("\n" + "=" * 80)
    
    columns_to_show = [
        'open_time', 'close', 'ema_fast', 'ema_slow', 'atr', 'rsi',
        'bb_upper', 'bb_lower', 'macd', 'macd_signal'
    ]
    
    print(df[columns_to_show].tail(10).to_string())
    
    # Show indicator statistics
    print("\n" + "=" * 80)
    print("4. Indicator Statistics:")
    print("=" * 80)
    
    print(f"\nEMA Fast (20):  {df['ema_fast'].iloc[-1]:.2f}")
    print(f"EMA Slow (50):  {df['ema_slow'].iloc[-1]:.2f}")
    print(f"EMA Long (200): {df['ema_long'].iloc[-1]:.2f}")
    print(f"\nATR:            {df['atr'].iloc[-1]:.4f}")
    print(f"RSI:            {df['rsi'].iloc[-1]:.2f}")
    print(f"\nMACD:           {df['macd'].iloc[-1]:.4f}")
    print(f"MACD Signal:    {df['macd_signal'].iloc[-1]:.4f}")
    print(f"MACD Histogram: {df['macd_histogram'].iloc[-1]:.4f}")
    
    # Check for crossovers
    print("\n" + "=" * 80)
    print("5. Signal Detection:")
    print("=" * 80)
    
    # EMA crossover
    if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]:
        if df['ema_fast'].iloc[-2] <= df['ema_slow'].iloc[-2]:
            print("\n✅ BULLISH: Fast EMA crossed above Slow EMA!")
        else:
            print("\n📈 Fast EMA is above Slow EMA (bullish)")
    else:
        if df['ema_fast'].iloc[-2] >= df['ema_slow'].iloc[-2]:
            print("\n❌ BEARISH: Fast EMA crossed below Slow EMA!")
        else:
            print("\n📉 Fast EMA is below Slow EMA (bearish)")
    
    # RSI conditions
    if df['rsi'].iloc[-1] > 70:
        print("⚠️  RSI indicates OVERBOUGHT condition")
    elif df['rsi'].iloc[-1] < 30:
        print("⚠️  RSI indicates OVERSOLD condition")
    else:
        print(f"✓  RSI is neutral ({df['rsi'].iloc[-1]:.1f})")
    
    # MACD crossover
    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
        if df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
            print("✅ BULLISH: MACD crossed above Signal!")
        else:
            print("📈 MACD is above Signal (bullish)")
    else:
        if df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
            print("❌ BEARISH: MACD crossed below Signal!")
        else:
            print("📉 MACD is below Signal (bearish)")
    
    print("\n" + "=" * 80)
    print("Indicator calculations completed!")
    print("=" * 80)
