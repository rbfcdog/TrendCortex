"""
Optimized Conservative Deterministic Strategies
Focus on capital preservation and high win rates
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import talib


class ConservativeBreakout:
    """
    Ultra-conservative breakout strategy
    Only trades confirmed multi-day breakouts with volume
    """
    
    def __init__(self):
        self.name = "Conservative Breakout"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Long-term resistance (50-day high)
        df['resistance_50'] = df['high'].rolling(50).max()
        
        # Breakout must be clean (above by 1%)
        df['clean_breakout'] = df['close'] > (df['resistance_50'].shift(1) * 1.01)
        
        # Volume must be 2x average
        df['volume_avg'] = df['volume'].rolling(50).mean()
        df['volume_spike'] = df['volume'] > (df['volume_avg'] * 2.0)
        
        # Must also be above 200 EMA (strong trend)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['strong_trend'] = df['close'] > df['ema_200']
        
        # RSI must be strong but not overbought
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_good'] = (df['rsi'] > 55) & (df['rsi'] < 75)
        
        # Entry: All conditions must be met
        df['signal'] = 0
        df.loc[
            df['clean_breakout'] &
            df['volume_spike'] &
            df['strong_trend'] &
            df['rsi_good'],
            'signal'
        ] = 1
        
        # Exit: Price closes below 20 EMA
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['exit_signal'] = 0
        df.loc[df['close'] < df['ema_20'], 'exit_signal'] = 1
        
        return df


class SwingHighLow:
    """
    Trade swing highs/lows with strict confirmation
    Wait for structure break + retest + confirmation candle
    """
    
    def __init__(self):
        self.name = "Swing High/Low"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Identify swing lows (20-bar lowest low)
        df['swing_low'] = df['low'].rolling(20, center=True).min()
        df['is_swing_low'] = df['low'] == df['swing_low']
        
        # Mark last swing low level
        df['last_swing_low'] = np.nan
        df.loc[df['is_swing_low'], 'last_swing_low'] = df['low']
        df['last_swing_low'] = df['last_swing_low'].ffill()
        
        # Price must break above swing low
        df['broke_high'] = df['close'] > df['last_swing_low'] * 1.02
        
        # Wait for retest (price comes back within 1% of swing low)
        df['retest'] = (df['low'] <= df['last_swing_low'] * 1.01) & \
                        (df['low'] >= df['last_swing_low'] * 0.99)
        
        # Confirmation candle (bullish engulfing after retest)
        df['bullish_candle'] = df['close'] > df['open']
        df['big_body'] = (df['close'] - df['open']) > (df['high'] - df['low']) * 0.6
        df['confirmation'] = df['bullish_candle'] & df['big_body']
        
        # Trend must be up (50 EMA rising)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['trend_up'] = df['ema_50'] > df['ema_50'].shift(5)
        
        # Entry: Retest + confirmation + uptrend
        df['signal'] = 0
        df.loc[
            df['retest'].shift(1) &  # Yesterday was retest
            df['confirmation'] &       # Today is confirmation
            df['trend_up'],
            'signal'
        ] = 1
        
        # Exit: Break below swing low or trend reversal
        df['exit_signal'] = 0
        df.loc[
            (df['close'] < df['last_swing_low'] * 0.98) |
            ~df['trend_up'],
            'exit_signal'
        ] = 1
        
        return df


class TrendStrengthFilter:
    """
    Only enter when trend is extremely strong
    Use ADX, EMA alignment, and momentum
    """
    
    def __init__(self):
        self.name = "Trend Strength Filter"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Multiple EMAs must be aligned
        df['ema_10'] = talib.EMA(df['close'], timeperiod=10)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_100'] = talib.EMA(df['close'], timeperiod=100)
        
        # Perfect alignment (all EMAs in order)
        df['ema_aligned'] = (
            (df['ema_10'] > df['ema_20']) &
            (df['ema_20'] > df['ema_50']) &
            (df['ema_50'] > df['ema_100'])
        )
        
        # ADX shows strong trend (>25)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['strong_adx'] = df['adx'] > 25
        
        # Price above all EMAs
        df['above_all'] = (
            (df['close'] > df['ema_10']) &
            (df['close'] > df['ema_20']) &
            (df['close'] > df['ema_50']) &
            (df['close'] > df['ema_100'])
        )
        
        # Pullback to 20 EMA (entry point)
        df['at_ema_20'] = (
            (df['low'] <= df['ema_20'] * 1.005) &
            (df['low'] >= df['ema_20'] * 0.995)
        )
        
        # Bounce confirmed
        df['bounce'] = (df['close'] > df['open']) & df['at_ema_20']
        
        # Entry: Strong trend + pullback + bounce
        df['signal'] = 0
        df.loc[
            df['ema_aligned'] &
            df['strong_adx'] &
            df['bounce'],
            'signal'
        ] = 1
        
        # Exit: EMA alignment breaks or ADX weakens
        df['exit_signal'] = 0
        df.loc[
            ~df['ema_aligned'] |
            (df['adx'] < 20),
            'exit_signal'
        ] = 1
        
        return df


class HighProbabilityPinbar:
    """
    Trade only perfect pinbar setups at key levels
    Must have perfect rejection wick and good risk/reward
    """
    
    def __init__(self):
        self.name = "High Probability Pinbar"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate candle components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Bullish pinbar criteria
        df['bullish_pinbar'] = (
            (df['lower_wick'] > df['body'] * 2) &  # Long lower wick
            (df['lower_wick'] > df['upper_wick'] * 2) &  # Minimal upper wick
            (df['lower_wick'] > df['total_range'] * 0.6) &  # Wick dominates
            (df['close'] > df['open'])  # Closes green
        )
        
        # Must be at support (50 EMA or previous swing low)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['at_support'] = (
            (df['low'] <= df['ema_50'] * 1.01) &
            (df['low'] >= df['ema_50'] * 0.99)
        )
        
        # Swing low support
        df['swing_low_20'] = df['low'].rolling(20).min()
        df['at_swing_low'] = (
            (df['low'] <= df['swing_low_20'].shift(1) * 1.01) &
            (df['low'] >= df['swing_low_20'].shift(1) * 0.99)
        )
        
        # Trend context (must be in uptrend)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['uptrend'] = df['close'] > df['ema_200']
        
        # Entry: Perfect pinbar at support in uptrend
        df['signal'] = 0
        df.loc[
            df['bullish_pinbar'] &
            (df['at_support'] | df['at_swing_low']) &
            df['uptrend'],
            'signal'
        ] = 1
        
        # Exit: Close below pinbar low or below 50 EMA
        df['pinbar_low'] = np.nan
        df.loc[df['signal'] == 1, 'pinbar_low'] = df['low']
        df['pinbar_low'] = df['pinbar_low'].ffill()
        
        df['exit_signal'] = 0
        df.loc[
            (df['close'] < df['pinbar_low']) |
            (df['close'] < df['ema_50']),
            'exit_signal'
        ] = 1
        
        return df


class VolumeProfileBreakout:
    """
    Use volume profile to find high-probability breakout points
    Trade breakouts from low-volume areas (points of control)
    """
    
    def __init__(self):
        self.name = "Volume Profile Breakout"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate volume by price level
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Low volume nodes (potential breakout points)
        df['low_volume'] = df['volume'] < (df['volume_ma'] * 0.5)
        
        # High volume nodes (support/resistance)
        df['high_volume'] = df['volume'] > (df['volume_ma'] * 2.0)
        
        # Consolidation (tight range + low volume)
        df['range_20'] = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['tight_range'] = df['range_20'] < (df['atr'] * 3)
        
        # Breakout from consolidation
        df['breakout'] = (
            (df['close'] > df['high'].rolling(20).max().shift(1)) &
            df['tight_range'].shift(1)
        )
        
        # Volume explosion on breakout
        df['volume_explosion'] = df['volume'] > (df['volume_ma'] * 3.0)
        
        # Trend alignment
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['with_trend'] = df['close'] > df['ema_50']
        
        # Entry: Breakout + volume + trend
        df['signal'] = 0
        df.loc[
            df['breakout'] &
            df['volume_explosion'] &
            df['with_trend'],
            'signal'
        ] = 1
        
        # Exit: Back into consolidation zone
        df['consolidation_high'] = df['high'].rolling(20).max()
        df['exit_signal'] = 0
        df.loc[
            df['close'] < df['consolidation_high'].shift(20),
            'exit_signal'
        ] = 1
        
        return df


def get_optimized_strategies():
    """Get all optimized conservative strategies"""
    return [
        ConservativeBreakout(),
        SwingHighLow(),
        TrendStrengthFilter(),
        HighProbabilityPinbar(),
        VolumeProfileBreakout()
    ]


if __name__ == "__main__":
    print("🎯 Optimized Conservative Strategies")
    print("=" * 80)
    for i, strategy in enumerate(get_optimized_strategies(), 1):
        print(f"{i}. {strategy.name}")
    print("=" * 80)
