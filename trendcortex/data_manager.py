"""
Data Manager Module

Handles fetching, caching, and transforming market data from the exchange.
Provides pandas DataFrames for technical analysis.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from cachetools import TTLCache

from trendcortex.api_client import WEEXAPIClient
from trendcortex.config import Config
from trendcortex.logger import get_logger
from trendcortex.utils import timestamp_to_datetime


class DataManager:
    """
    Manages market data fetching, caching, and transformation.
    
    Provides clean pandas DataFrames with OHLCV data, orderbook snapshots,
    and recent trades for technical analysis.
    """
    
    def __init__(self, api_client: WEEXAPIClient, config: Config):
        """
        Initialize data manager.
        
        Args:
            api_client: WEEX API client instance
            config: System configuration
        """
        self.api = api_client
        self.config = config
        self.logger = get_logger()
        
        # Cache configuration
        self.cache_enabled = config.data.cache_enabled
        self.cache_ttl = config.data.cache_ttl_seconds
        
        # In-memory caches
        self._candle_cache: Dict[str, TTLCache] = {}
        self._ticker_cache = TTLCache(maxsize=100, ttl=self.cache_ttl)
        self._orderbook_cache = TTLCache(maxsize=50, ttl=5)  # 5 second TTL for orderbook
        
        # Contract specifications cache
        self._contract_specs: Dict[str, Dict] = {}
    
    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 100,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch candlestick data and return as pandas DataFrame.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles to fetch
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if use_cache and self.cache_enabled:
            if cache_key in self._candle_cache:
                cached_df = self._candle_cache[cache_key].get("data")
                if cached_df is not None and len(cached_df) >= limit:
                    self.logger.debug(f"Using cached candle data for {cache_key}")
                    return cached_df.tail(limit).copy()
        
        # Fetch from API
        try:
            raw_candles = await self.api.get_candles(
                symbol=symbol,
                interval=timeframe,
                limit=limit,
            )
            
            if not raw_candles:
                self.logger.warning(f"No candle data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Transform to DataFrame
            df = self._parse_candles(raw_candles)
            
            # Cache the result
            if self.cache_enabled:
                if cache_key not in self._candle_cache:
                    self._candle_cache[cache_key] = TTLCache(maxsize=1, ttl=self.cache_ttl)
                self._candle_cache[cache_key]["data"] = df
            
            self.logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch candles for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _parse_candles(self, raw_candles: List) -> pd.DataFrame:
        """
        Parse raw candle data into pandas DataFrame.
        
        Args:
            raw_candles: Raw candle data from API
            
        Returns:
            Formatted DataFrame
        """
        # Expected format: [timestamp, open, high, low, close, volume, ...]
        if not raw_candles:
            return pd.DataFrame()
        
        # Handle different API response formats
        if isinstance(raw_candles[0], list):
            # Array format
            df = pd.DataFrame(raw_candles, columns=[
                "timestamp", "open", "high", "low", "close", "volume"
            ])
        elif isinstance(raw_candles[0], dict):
            # Dict format
            df = pd.DataFrame(raw_candles)
        else:
            self.logger.error(f"Unknown candle format: {type(raw_candles[0])}")
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Missing required columns in candle data: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df[required_cols]
    
    async def get_current_price(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            use_cache: Whether to use cached data
            
        Returns:
            Current price or None if unavailable
        """
        cache_key = f"price_{symbol}"
        
        # Check cache
        if use_cache and self.cache_enabled:
            cached_price = self._ticker_cache.get(cache_key)
            if cached_price is not None:
                return cached_price
        
        # Fetch from API
        try:
            ticker = await self.api.get_ticker(symbol)
            price = float(ticker.get("last", 0))
            
            # Cache the result
            if self.cache_enabled:
                self._ticker_cache[cache_key] = price
            
            return price
            
        except Exception as e:
            self.logger.error(f"Failed to fetch price for {symbol}: {e}")
            return None
    
    async def get_ticker_data(self, symbol: str) -> Optional[Dict]:
        """
        Get full ticker data including 24h stats.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data dictionary
        """
        try:
            ticker = await self.api.get_ticker(symbol)
            return {
                "symbol": ticker.get("symbol"),
                "last": float(ticker.get("last", 0)),
                "bid": float(ticker.get("best_bid", 0)),
                "ask": float(ticker.get("best_ask", 0)),
                "high_24h": float(ticker.get("high_24h", 0)),
                "low_24h": float(ticker.get("low_24h", 0)),
                "volume_24h": float(ticker.get("volume_24h", 0)),
                "price_change_percent": float(ticker.get("priceChangePercent", 0)),
                "mark_price": float(ticker.get("markPrice", 0)),
                "index_price": float(ticker.get("indexPrice", 0)),
                "timestamp": int(ticker.get("timestamp", 0)),
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return None
    
    async def get_orderbook(
        self,
        symbol: str,
        depth: int = 20,
        use_cache: bool = True,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get orderbook data.
        
        Args:
            symbol: Trading pair symbol
            depth: Number of price levels
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with 'bids' and 'asks' DataFrames
        """
        cache_key = f"orderbook_{symbol}_{depth}"
        
        # Check cache
        if use_cache and self.cache_enabled:
            cached_book = self._orderbook_cache.get(cache_key)
            if cached_book is not None:
                return cached_book
        
        # Fetch from API
        try:
            raw_book = await self.api.get_orderbook(symbol, depth)
            
            # Parse bids and asks
            bids_df = pd.DataFrame(raw_book.get("bids", []), columns=["price", "size"])
            asks_df = pd.DataFrame(raw_book.get("asks", []), columns=["price", "size"])
            
            bids_df["price"] = pd.to_numeric(bids_df["price"])
            bids_df["size"] = pd.to_numeric(bids_df["size"])
            asks_df["price"] = pd.to_numeric(asks_df["price"])
            asks_df["size"] = pd.to_numeric(asks_df["size"])
            
            orderbook = {
                "bids": bids_df,
                "asks": asks_df,
                "timestamp": raw_book.get("timestamp", int(datetime.now().timestamp() * 1000)),
            }
            
            # Cache the result
            if self.cache_enabled:
                self._orderbook_cache[cache_key] = orderbook
            
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return None
    
    async def get_contract_specs(self, symbol: str) -> Optional[Dict]:
        """
        Get contract specifications (precision, limits, etc.).
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Contract specifications dictionary
        """
        # Check cache
        if symbol in self._contract_specs:
            return self._contract_specs[symbol]
        
        # Fetch from API
        try:
            contract_info = await self.api.get_contract_info(symbol)
            
            specs = {
                "symbol": contract_info.get("symbol"),
                "contract_val": float(contract_info.get("contract_val", 1)),
                "tick_size": float(contract_info.get("tick_size", 1)),
                "size_increment": float(contract_info.get("size_increment", 1)),
                "min_order_size": float(contract_info.get("minOrderSize", 0.0001)),
                "max_order_size": float(contract_info.get("maxOrderSize", 1000)),
                "max_position_size": float(contract_info.get("maxPositionSize", 100000)),
                "min_leverage": int(contract_info.get("minLeverage", 1)),
                "max_leverage": int(contract_info.get("maxLeverage", 400)),
                "maker_fee": float(contract_info.get("makerFeeRate", 0.0002)),
                "taker_fee": float(contract_info.get("takerFeeRate", 0.0008)),
            }
            
            # Cache permanently (specs don't change often)
            self._contract_specs[symbol] = specs
            
            return specs
            
        except Exception as e:
            self.logger.error(f"Failed to fetch contract specs for {symbol}: {e}")
            return None
    
    async def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str],
        limit: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch candle data for multiple timeframes concurrently.
        
        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframe intervals
            limit: Number of candles per timeframe
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        tasks = [
            self.get_candles(symbol, tf, limit)
            for tf in timeframes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            tf: result if not isinstance(result, Exception) else pd.DataFrame()
            for tf, result in zip(timeframes, results)
        }
    
    async def get_multiple_symbols_data(
        self,
        symbols: List[str],
        timeframe: str = "5m",
        limit: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch candle data for multiple symbols concurrently.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe interval
            limit: Number of candles
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        tasks = [
            self.get_candles(symbol, timeframe, limit)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result if not isinstance(result, Exception) else pd.DataFrame()
            for symbol, result in zip(symbols, results)
        }
    
    def calculate_vwap(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate Volume Weighted Average Price.
        
        Args:
            df: DataFrame with OHLCV data
            window: Rolling window size
            
        Returns:
            VWAP series
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).rolling(window).sum() / df["volume"].rolling(window).sum()
        return vwap
    
    def calculate_returns(self, df: pd.DataFrame, periods: int = 1) -> pd.Series:
        """
        Calculate price returns.
        
        Args:
            df: DataFrame with price data
            periods: Number of periods for return calculation
            
        Returns:
            Returns series
        """
        return df["close"].pct_change(periods=periods)
    
    def resample_candles(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample candles to different timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            target_timeframe: Target timeframe (e.g., "15m", "1h")
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        df_copy.set_index("timestamp", inplace=True)
        
        # Define aggregation rules
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        
        # Resample
        resampled = df_copy.resample(target_timeframe).agg(agg_dict).dropna()
        resampled.reset_index(inplace=True)
        
        return resampled
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._candle_cache.clear()
        self._ticker_cache.clear()
        self._orderbook_cache.clear()
        self.logger.info("Data cache cleared")
