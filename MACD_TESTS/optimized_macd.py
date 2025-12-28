"""
Optimize Original MACD - Parameter Tuning
Find the sweet spot that makes it profitable
"""

import numpy as np
import pandas as pd
import talib


class OptimizedMACD:
    """
    Take the exact original logic, just tune parameters
    Original: -0.23%, 35.9% WR, PF 1.09
    """
    
    def __init__(
        self,
        trend_period: int = 200,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        require_macd_positive: bool = True
    ):
        self.name = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}_T{trend_period}"
        self.trend_period = trend_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.require_macd_positive = require_macd_positive
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Trend
        df['trend_ema'] = talib.EMA(df['close'], timeperiod=self.trend_period)
        df['in_uptrend'] = df['close'] > df['trend_ema']
        
        # MACD
        macd, signal, hist = talib.MACD(
            df['close'],
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        
        # Crossover
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # Entry
        df['signal'] = 0
        if self.require_macd_positive:
            df.loc[
                df['in_uptrend'] & 
                df['macd_cross_up'] & 
                (df['macd'] > 0),
                'signal'
            ] = 1
        else:
            df.loc[
                df['in_uptrend'] & 
                df['macd_cross_up'],
                'signal'
            ] = 1
        
        # Exit
        df['exit_signal'] = 0
        df.loc[
            df['macd_cross_down'] | 
            ~df['in_uptrend'],
            'exit_signal'
        ] = 1
        
        return df


class MACDWithVolumeFilter:
    """Add just volume filter to original MACD"""
    
    def __init__(self):
        self.name = "MACD + Volume"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Original MACD logic
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['in_uptrend'] = df['close'] > df['ema_200']
        
        macd, signal, hist = talib.MACD(df['close'], 12, 26, 9)
        df['macd'] = macd
        df['macd_signal'] = signal
        
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        # ADD: Simple volume filter
        df['volume_avg'] = df['volume'].rolling(20).mean()
        df['above_avg_volume'] = df['volume'] > df['volume_avg']
        
        # Entry: Original + volume
        df['signal'] = 0
        df.loc[
            df['in_uptrend'] & 
            df['macd_cross_up'] & 
            (df['macd'] > 0) &
            df['above_avg_volume'],  # NEW
            'signal'
        ] = 1
        
        # Exit: Same as original
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        df['exit_signal'] = 0
        df.loc[
            df['macd_cross_down'] | 
            ~df['in_uptrend'],
            'exit_signal'
        ] = 1
        
        return df


class MACDWithRSIFilter:
    """Add just RSI filter to original MACD"""
    
    def __init__(self):
        self.name = "MACD + RSI"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Original MACD
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['in_uptrend'] = df['close'] > df['ema_200']
        
        macd, signal, hist = talib.MACD(df['close'], 12, 26, 9)
        df['macd'] = macd
        df['macd_signal'] = signal
        
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        # ADD: RSI filter
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_bullish'] = (df['rsi'] > 50) & (df['rsi'] < 70)
        
        # Entry
        df['signal'] = 0
        df.loc[
            df['in_uptrend'] & 
            df['macd_cross_up'] & 
            (df['macd'] > 0) &
            df['rsi_bullish'],  # NEW
            'signal'
        ] = 1
        
        # Exit
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        df['exit_signal'] = 0
        df.loc[
            df['macd_cross_down'] | 
            ~df['in_uptrend'],
            'exit_signal'
        ] = 1
        
        return df


class MACDWith50EMA:
    """Add 50 EMA to original MACD for better trend"""
    
    def __init__(self):
        self.name = "MACD + 50/200 EMA"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Stronger trend: Both 50 and 200 EMA
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['strong_uptrend'] = (df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
        
        # Original MACD
        macd, signal, hist = talib.MACD(df['close'], 12, 26, 9)
        df['macd'] = macd
        df['macd_signal'] = signal
        
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        # Entry: Stronger trend requirement
        df['signal'] = 0
        df.loc[
            df['strong_uptrend'] &  # CHANGED
            df['macd_cross_up'] & 
            (df['macd'] > 0),
            'signal'
        ] = 1
        
        # Exit
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        df['exit_signal'] = 0
        df.loc[
            df['macd_cross_down'] | 
            ~df['strong_uptrend'],  # CHANGED
            'exit_signal'
        ] = 1
        
        return df


def get_macd_variants():
    """Get MACD parameter variations"""
    variants = []
    
    # Original parameters
    variants.append(OptimizedMACD(200, 12, 26, 9, True))
    
    # Try different trend periods
    variants.append(OptimizedMACD(100, 12, 26, 9, True))  # Faster trend
    variants.append(OptimizedMACD(50, 12, 26, 9, True))   # Even faster
    
    # Try faster MACD
    variants.append(OptimizedMACD(200, 8, 17, 9, True))   # Faster MACD
    variants.append(OptimizedMACD(200, 5, 13, 5, True))   # Much faster
    
    # Try slower MACD
    variants.append(OptimizedMACD(200, 16, 32, 9, True))  # Slower MACD
    
    # Try without requiring MACD > 0
    variants.append(OptimizedMACD(200, 12, 26, 9, False))
    
    # Try with filters
    variants.append(MACDWithVolumeFilter())
    variants.append(MACDWithRSIFilter())
    variants.append(MACDWith50EMA())
    
    return variants


if __name__ == "__main__":
    print("🎯 MACD Parameter Optimization")
    print("=" * 80)
    print(f"\nTesting {len(get_macd_variants())} variants")
    print("=" * 80)
