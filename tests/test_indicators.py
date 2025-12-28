"""
Unit tests for Technical Indicators module.
"""

import pytest
import pandas as pd
import numpy as np

from trendcortex.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    apply_all_indicators,
)


class TestIndicators:
    """Test suite for technical indicators"""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data"""
        return pd.Series(range(100, 200))
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100),
        })
    
    def test_calculate_ema(self, sample_prices):
        """Test EMA calculation"""
        ema = calculate_ema(sample_prices, period=20)
        
        assert len(ema) == len(sample_prices)
        assert not ema.isna().all()
        # EMA should smooth the data
        assert ema.std() < sample_prices.std()
    
    def test_calculate_rsi(self, sample_prices):
        """Test RSI calculation"""
        rsi = calculate_rsi(sample_prices, period=14)
        
        assert len(rsi) == len(sample_prices)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_calculate_atr(self, sample_ohlcv):
        """Test ATR calculation"""
        atr = calculate_atr(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close'],
            period=14
        )
        
        assert len(atr) == len(sample_ohlcv)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
    
    def test_calculate_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands calculation"""
        middle, upper, lower = calculate_bollinger_bands(sample_prices, period=20, std_dev=2.0)
        
        assert len(middle) == len(sample_prices)
        assert len(upper) == len(sample_prices)
        assert len(lower) == len(sample_prices)
        
        # Upper band should be above middle, lower below
        valid_data = ~middle.isna()
        assert (upper[valid_data] > middle[valid_data]).all()
        assert (lower[valid_data] < middle[valid_data]).all()
    
    def test_calculate_macd(self, sample_prices):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = calculate_macd(
            sample_prices,
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        assert len(macd_line) == len(sample_prices)
        assert len(signal_line) == len(sample_prices)
        assert len(histogram) == len(sample_prices)
        
        # Histogram should be difference between MACD and signal
        valid_data = ~macd_line.isna() & ~signal_line.isna()
        np.testing.assert_array_almost_equal(
            histogram[valid_data],
            (macd_line - signal_line)[valid_data]
        )
    
    def test_apply_all_indicators(self, sample_ohlcv):
        """Test applying all indicators to dataframe"""
        df_with_indicators = apply_all_indicators(sample_ohlcv)
        
        # Check that indicator columns were added
        expected_columns = [
            'ema_fast', 'ema_slow', 'ema_trend',
            'rsi', 'atr', 'bb_upper', 'bb_lower',
            'macd', 'macd_signal', 'macd_hist'
        ]
        
        for col in expected_columns:
            assert col in df_with_indicators.columns
    
    def test_indicators_handle_empty_data(self):
        """Test that indicators handle empty data gracefully"""
        empty_series = pd.Series([])
        
        ema = calculate_ema(empty_series, period=20)
        assert len(ema) == 0
        
        rsi = calculate_rsi(empty_series, period=14)
        assert len(rsi) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
