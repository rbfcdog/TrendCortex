"""
Simple Binance Data Fetcher for Backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional


class BinanceDataFetcher:
    """Fetch historical data from Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical kline/candlestick data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            # Binance API endpoint
            url = f"{self.base_url}/klines"
            
            all_klines = []
            current_start = start_ms
            
            # Fetch in chunks (Binance has 1000 candle limit per request)
            while current_start < end_ms:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ms,
                    'limit': 1000
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"Error fetching data: {response.status_code}")
                    return None
                
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Update start time for next batch
                current_start = klines[-1][0] + 1
            
            if not all_klines:
                print("No data received from Binance")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Keep only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"Exception fetching data: {e}")
            return None


if __name__ == "__main__":
    # Test the fetcher
    fetcher = BinanceDataFetcher()
    df = fetcher.get_historical_klines("BTCUSDT", "1h", 7)
    
    if df is not None:
        print(f"Fetched {len(df)} candles")
        print(df.head())
        print(df.tail())
    else:
        print("Failed to fetch data")
