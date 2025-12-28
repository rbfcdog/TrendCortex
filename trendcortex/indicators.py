"""
Technical Indicators Module

Implements common technical analysis indicators using pandas/numpy.
Provides functions for EMA, RSI, ATR, Bollinger Bands, MACD, and more.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def calculate_ema(
    data: pd.Series,
    period: int,
    adjust: bool = False,
) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: Price series (typically close prices)
        period: EMA period
        adjust: Use adjusted EMA calculation
        
    Returns:
        EMA series
    """
    return data.ewm(span=period, adjust=adjust).mean()


def calculate_sma(
    data: pd.Series,
    period: int,
) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: Price series
        period: SMA period
        
    Returns:
        SMA series
    """
    return data.rolling(window=period).mean()


def calculate_rsi(
    data: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: Price series (typically close prices)
        period: RSI period (default: 14)
        
    Returns:
        RSI series (0-100)
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default: 14)
        
    Returns:
        ATR series
    """
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR using EMA
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price series (typically close prices)
        period: Period for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle_band = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return middle_band, upper_band, lower_band


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price series (typically close prices)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    
    return k, d


def calculate_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        OBV series
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) and directional indicators.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ADX period (default: 14)
        
    Returns:
        Tuple of (adx, plus_di, minus_di)
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=high.index)
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=low.index)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di


def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 20,
) -> pd.DataFrame:
    """
    Calculate volume profile (volume distribution across price levels).
    
    Args:
        df: DataFrame with OHLCV data
        bins: Number of price bins
        
    Returns:
        DataFrame with price levels and corresponding volumes
    """
    price_min = df["low"].min()
    price_max = df["high"].max()
    
    # Create price bins
    price_bins = np.linspace(price_min, price_max, bins + 1)
    
    # Assign each candle to a price bin
    df_copy = df.copy()
    df_copy["price_bin"] = pd.cut(
        df_copy["close"],
        bins=price_bins,
        labels=range(bins),
        include_lowest=True,
    )
    
    # Aggregate volume by price bin
    volume_profile = df_copy.groupby("price_bin")["volume"].sum()
    
    # Get midpoint of each bin
    bin_midpoints = [(price_bins[i] + price_bins[i+1]) / 2 for i in range(bins)]
    
    profile_df = pd.DataFrame({
        "price_level": bin_midpoints,
        "volume": volume_profile.values,
    })
    
    return profile_df


def calculate_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    num_levels: int = 5,
) -> Tuple[list, list]:
    """
    Identify support and resistance levels using local minima/maxima.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window for identifying local extrema
        num_levels: Number of levels to return
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    # Find local minima (support)
    local_min = df["low"].rolling(window=window, center=True).min()
    support_mask = df["low"] == local_min
    support_levels = df.loc[support_mask, "low"].tolist()
    
    # Find local maxima (resistance)
    local_max = df["high"].rolling(window=window, center=True).max()
    resistance_mask = df["high"] == local_max
    resistance_levels = df.loc[resistance_mask, "high"].tolist()
    
    # Remove duplicates and sort
    support_levels = sorted(set(support_levels))[-num_levels:]
    resistance_levels = sorted(set(resistance_levels), reverse=True)[:num_levels]
    
    return support_levels, resistance_levels


def calculate_volatility(
    data: pd.Series,
    period: int = 20,
    annualize: bool = False,
) -> pd.Series:
    """
    Calculate price volatility (standard deviation of returns).
    
    Args:
        data: Price series
        period: Period for calculation
        annualize: Whether to annualize volatility
        
    Returns:
        Volatility series
    """
    returns = data.pct_change()
    volatility = returns.rolling(window=period).std()
    
    if annualize:
        # Assume 365 days/year for crypto (24/7 market)
        volatility = volatility * np.sqrt(365)
    
    return volatility


def calculate_momentum(
    data: pd.Series,
    period: int = 10,
) -> pd.Series:
    """
    Calculate price momentum.
    
    Args:
        data: Price series
        period: Period for momentum calculation
        
    Returns:
        Momentum series
    """
    return data.diff(period)


def detect_divergence(
    price: pd.Series,
    indicator: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    Detect bullish/bearish divergence between price and indicator.
    
    Args:
        price: Price series
        indicator: Indicator series (e.g., RSI, MACD)
        window: Window for comparing trends
        
    Returns:
        Series with divergence signals (1=bullish, -1=bearish, 0=none)
    """
    # Calculate trends
    price_trend = price.diff(window)
    indicator_trend = indicator.diff(window)
    
    # Detect divergence
    divergence = pd.Series(0, index=price.index)
    
    # Bullish divergence: price making lower lows, indicator making higher lows
    bullish = (price_trend < 0) & (indicator_trend > 0)
    divergence[bullish] = 1
    
    # Bearish divergence: price making higher highs, indicator making lower highs
    bearish = (price_trend > 0) & (indicator_trend < 0)
    divergence[bearish] = -1
    
    return divergence


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert regular candles to Heikin-Ashi candles.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with Heikin-Ashi candles
    """
    ha_df = df.copy()
    
    # Heikin-Ashi close
    ha_df["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    
    # Heikin-Ashi open
    ha_df["open"] = (df["open"].shift(1) + df["close"].shift(1)) / 2
    ha_df.iloc[0, ha_df.columns.get_loc("open")] = df.iloc[0]["open"]
    
    # Heikin-Ashi high and low
    ha_df["high"] = ha_df[["open", "close", "high"]].max(axis=1)
    ha_df["low"] = ha_df[["open", "close", "low"]].min(axis=1)
    
    return ha_df


def apply_all_indicators(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply all common technical indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dict with indicator parameters
        
    Returns:
        DataFrame with all indicator columns added
    """
    if config is None:
        config = {}
    
    df = df.copy()
    
    # Moving averages
    ema_fast = config.get("ema_fast_period", 12)
    ema_slow = config.get("ema_slow_period", 26)
    ema_trend = config.get("ema_trend_period", 200)
    
    df["ema_fast"] = calculate_ema(df["close"], ema_fast)
    df["ema_slow"] = calculate_ema(df["close"], ema_slow)
    df["ema_trend"] = calculate_ema(df["close"], ema_trend)
    df["sma_20"] = calculate_sma(df["close"], 20)
    
    # RSI
    rsi_period = config.get("rsi_period", 14)
    df["rsi"] = calculate_rsi(df["close"], rsi_period)
    
    # ATR
    atr_period = config.get("atr_period", 14)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], atr_period)
    df["atr_percent"] = (df["atr"] / df["close"]) * 100
    
    # Bollinger Bands
    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.0)
    df["bb_middle"], df["bb_upper"], df["bb_lower"] = calculate_bollinger_bands(
        df["close"], bb_period, bb_std
    )
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # MACD
    macd_fast = config.get("macd_fast", 12)
    macd_slow = config.get("macd_slow", 26)
    macd_signal = config.get("macd_signal", 9)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(
        df["close"], macd_fast, macd_slow, macd_signal
    )
    
    # Stochastic
    df["stoch_k"], df["stoch_d"] = calculate_stochastic(
        df["high"], df["low"], df["close"]
    )
    
    # Volume indicators
    df["obv"] = calculate_obv(df["close"], df["volume"])
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    
    # Volatility
    df["volatility"] = calculate_volatility(df["close"], period=20)
    
    # Momentum
    df["momentum"] = calculate_momentum(df["close"], period=10)
    
    return df
