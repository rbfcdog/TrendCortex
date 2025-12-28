"""
Historical Data Fetcher for Binance

This module provides functions to download historical OHLCV (Open, High, Low, Close, Volume)
data from Binance public API without requiring authentication. It handles batching for
fetching data beyond the 1000-bar limit and saves data to CSV files for caching.

Key Features:
- Fetches data from Binance public REST API
- Handles pagination to fetch unlimited historical data
- Caches data to CSV files to avoid redundant API calls
- Time conversion utilities (datetime <-> milliseconds)
- No API keys required (public endpoint)
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging

from config import (
    BINANCE_KLINES_ENDPOINT,
    MAX_CANDLES_PER_REQUEST,
    REQUEST_DELAY_SECONDS,
    DATA_DIR,
    validate_symbol,
    validate_interval,
    interval_to_milliseconds,
    get_data_path,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TIME CONVERSION UTILITIES
# =============================================================================

def datetime_to_ms(dt: datetime) -> int:
    """
    Convert datetime object to milliseconds timestamp.
    
    Args:
        dt: Python datetime object
    
    Returns:
        Timestamp in milliseconds
    
    Example:
        >>> dt = datetime(2024, 1, 1, 0, 0, 0)
        >>> ms = datetime_to_ms(dt)
        >>> print(ms)
        1704067200000
    """
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """
    Convert milliseconds timestamp to datetime object.
    
    Args:
        ms: Timestamp in milliseconds
    
    Returns:
        Python datetime object
    
    Example:
        >>> ms = 1704067200000
        >>> dt = ms_to_datetime(ms)
        >>> print(dt)
        2024-01-01 00:00:00
    """
    return datetime.fromtimestamp(ms / 1000)


# =============================================================================
# BINANCE API FUNCTIONS
# =============================================================================

def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    limit: int = MAX_CANDLES_PER_REQUEST
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance for a specific time range.
    
    This function makes a single API call to Binance to fetch up to 1000 candles.
    For larger date ranges, use fetch_ohlcv_batch() which handles pagination.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Timeframe (e.g., "1h", "15m")
        start_ts: Start timestamp in milliseconds
        end_ts: End timestamp in milliseconds
        limit: Maximum number of candles to fetch (default: 1000)
    
    Returns:
        DataFrame with columns: [open_time, open, high, low, close, volume,
                                  close_time, quote_volume, trades, taker_buy_base,
                                  taker_buy_quote, ignore]
    
    Raises:
        ValueError: If symbol or interval is invalid
        requests.RequestException: If API call fails
    
    Example:
        >>> start = datetime_to_ms(datetime(2024, 1, 1))
        >>> end = datetime_to_ms(datetime(2024, 1, 2))
        >>> df = fetch_ohlcv("BTCUSDT", "1h", start, end)
        >>> print(df.head())
    """
    # Validate inputs
    if not validate_symbol(symbol):
        raise ValueError(f"Invalid symbol: {symbol}. Must be one of the approved symbols.")
    
    if not validate_interval(interval):
        raise ValueError(f"Invalid interval: {interval}")
    
    # Build request parameters
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': limit
    }
    
    logger.debug(f"Fetching {symbol} {interval} from {ms_to_datetime(start_ts)} to {ms_to_datetime(end_ts)}")
    
    try:
        # Make API request
        response = requests.get(BINANCE_KLINES_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        if not data:
            logger.warning(f"No data returned for {symbol} {interval}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df['trades'] = df['trades'].astype(int)
        
        logger.debug(f"Fetched {len(df)} candles")
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


def fetch_ohlcv_batch(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a large date range by batching multiple API calls.
    
    Binance limits each API call to 1000 candles. This function automatically
    splits the date range into multiple batches and combines the results.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Timeframe (e.g., "1h", "15m")
        start_date: Start date (datetime object)
        end_date: End date (datetime object)
    
    Returns:
        Combined DataFrame with all OHLCV data for the date range
    
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 12, 31)
        >>> df = fetch_ohlcv_batch("BTCUSDT", "1h", start, end)
        >>> print(f"Fetched {len(df)} candles")
    """
    logger.info(f"Fetching {symbol} {interval} from {start_date.date()} to {end_date.date()}")
    
    # Convert dates to milliseconds
    start_ts = datetime_to_ms(start_date)
    end_ts = datetime_to_ms(end_date)
    
    # Calculate interval in milliseconds
    interval_ms = interval_to_milliseconds(interval)
    
    # Calculate how many candles we can fetch per batch
    candles_per_batch = MAX_CANDLES_PER_REQUEST
    batch_duration_ms = candles_per_batch * interval_ms
    
    all_data = []
    current_start = start_ts
    batch_count = 0
    
    # Fetch data in batches
    while current_start < end_ts:
        # Calculate end of this batch
        current_end = min(current_start + batch_duration_ms, end_ts)
        
        try:
            # Fetch batch
            df_batch = fetch_ohlcv(symbol, interval, current_start, current_end)
            
            if not df_batch.empty:
                all_data.append(df_batch)
                batch_count += 1
                
                # Get the last timestamp from this batch
                last_timestamp = df_batch['close_time'].iloc[-1]
                current_start = datetime_to_ms(last_timestamp) + 1
                
                logger.info(f"Batch {batch_count}: Fetched {len(df_batch)} candles "
                          f"(up to {last_timestamp})")
            else:
                # No more data available
                logger.warning(f"No data returned for batch starting at {ms_to_datetime(current_start)}")
                break
            
            # Rate limiting: small delay between requests
            time.sleep(REQUEST_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            # Continue with next batch on error
            current_start = current_end + 1
            continue
    
    # Combine all batches
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates (can happen at batch boundaries)
        combined_df = combined_df.drop_duplicates(subset=['open_time'], keep='first')
        
        # Sort by time
        combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
        
        logger.info(f"Successfully fetched {len(combined_df)} total candles in {batch_count} batches")
        
        return combined_df
    else:
        logger.warning("No data fetched")
        return pd.DataFrame()


# =============================================================================
# CSV STORAGE FUNCTIONS
# =============================================================================

def save_to_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        path: File path (Path object or string)
    
    Example:
        >>> df = fetch_ohlcv_batch("BTCUSDT", "1h", start_date, end_date)
        >>> save_to_csv(df, "data/BTCUSDT_1h.csv")
    """
    path = Path(path)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(path, index=False)
    
    logger.info(f"Saved {len(df)} rows to {path}")


def load_from_csv(path: Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        path: File path (Path object or string)
    
    Returns:
        DataFrame loaded from CSV
    
    Raises:
        FileNotFoundError: If file doesn't exist
    
    Example:
        >>> df = load_from_csv("data/BTCUSDT_1h.csv")
        >>> print(df.head())
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load CSV
    df = pd.read_csv(path)
    
    # Convert datetime columns
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'])
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'])
    
    logger.info(f"Loaded {len(df)} rows from {path}")
    
    return df


def check_cache(symbol: str, interval: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Check if data is already cached locally.
    
    Args:
        symbol: Trading pair
        interval: Timeframe
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame if cache exists and is valid, None otherwise
    """
    cache_path = get_data_path(symbol, interval, start_date, end_date)
    
    if cache_path.exists():
        try:
            df = load_from_csv(cache_path)
            logger.info(f"Using cached data from {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    return None


# =============================================================================
# HIGH-LEVEL CONVENIENCE FUNCTION
# =============================================================================

def get_historical_data(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Get historical OHLCV data with automatic caching.
    
    This is the main function you should use to get historical data.
    It automatically:
    1. Checks if data is cached locally
    2. Fetches from Binance if not cached or force_refresh=True
    3. Saves to cache for future use
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Timeframe (e.g., "1h", "15m")
        start_date: Start date
        end_date: End date
        use_cache: Whether to use cached data if available (default: True)
        force_refresh: Force re-fetch even if cached (default: False)
    
    Returns:
        DataFrame with OHLCV data
    
    Example:
        >>> from datetime import datetime, timedelta
        >>> end = datetime.now()
        >>> start = end - timedelta(days=30)
        >>> df = get_historical_data("BTCUSDT", "1h", start, end)
        >>> print(df.head())
    """
    # Check cache first (unless force refresh)
    if use_cache and not force_refresh:
        cached_df = check_cache(symbol, interval, start_date, end_date)
        if cached_df is not None:
            return cached_df
    
    # Fetch from API
    logger.info(f"Fetching fresh data for {symbol} {interval}")
    df = fetch_ohlcv_batch(symbol, interval, start_date, end_date)
    
    if df.empty:
        logger.warning("No data fetched")
        return df
    
    # Save to cache
    cache_path = get_data_path(symbol, interval, start_date, end_date)
    save_to_csv(df, cache_path)
    
    return df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the data fetcher module.
    
    Run this file directly to test fetching data:
        python data_fetcher.py
    """
    
    # Example 1: Fetch last 7 days of hourly data for Bitcoin
    print("=" * 80)
    print("Example 1: Fetch BTC 1h data for last 7 days")
    print("=" * 80)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    btc_df = get_historical_data("BTCUSDT", "1h", start_date, end_date)
    
    print(f"\nFetched {len(btc_df)} candles")
    print("\nFirst 5 rows:")
    print(btc_df[['open_time', 'open', 'high', 'low', 'close', 'volume']].head())
    print("\nLast 5 rows:")
    print(btc_df[['open_time', 'open', 'high', 'low', 'close', 'volume']].tail())
    
    # Example 2: Fetch 15-minute data for multiple symbols
    print("\n" + "=" * 80)
    print("Example 2: Fetch 15m data for ETH and SOL")
    print("=" * 80)
    
    symbols = ["ETHUSDT", "SOLUSDT"]
    
    for symbol in symbols:
        print(f"\nFetching {symbol}...")
        df = get_historical_data(symbol, "15m", start_date, end_date)
        print(f"  - Fetched {len(df)} candles")
        print(f"  - Date range: {df['open_time'].min()} to {df['open_time'].max()}")
        print(f"  - Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # Example 3: Demonstrate caching
    print("\n" + "=" * 80)
    print("Example 3: Demonstrate caching (second call should be instant)")
    print("=" * 80)
    
    print("\nFirst call (will fetch from API):")
    start_time = time.time()
    df1 = get_historical_data("BTCUSDT", "1h", start_date, end_date)
    time1 = time.time() - start_time
    print(f"  - Time taken: {time1:.2f} seconds")
    
    print("\nSecond call (should use cache):")
    start_time = time.time()
    df2 = get_historical_data("BTCUSDT", "1h", start_date, end_date)
    time2 = time.time() - start_time
    print(f"  - Time taken: {time2:.2f} seconds")
    print(f"  - Speed improvement: {time1/time2:.1f}x faster")
    
    # Example 4: Time conversion utilities
    print("\n" + "=" * 80)
    print("Example 4: Time conversion utilities")
    print("=" * 80)
    
    now = datetime.now()
    now_ms = datetime_to_ms(now)
    back_to_dt = ms_to_datetime(now_ms)
    
    print(f"\nOriginal datetime: {now}")
    print(f"Converted to ms:   {now_ms}")
    print(f"Converted back:    {back_to_dt}")
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
