"""
Refined MACD Strategy - Push to Profitability
Take the best performer (-0.23%) and optimize it
"""

import numpy as np
import pandas as pd
import talib


class RefinedMACDTrend:
    """
    Original MACD: -0.23%, 35.9% WR, PF 1.09 (CLOSEST TO PROFIT!)
    
    Refinements to push into profitability:
    1. Add volume filter (institutions must be buying)
    2. Stricter trend requirements (stronger EMA alignment)
    3. Wait for MACD pullback (better entry timing)
    4. Add RSI filter (momentum confirmation)
    5. Wider take profit (let winners run more)
    """
    
    def __init__(self):
        self.name = "Refined MACD Trend"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Original: 200 EMA trend
        # Refined: Add 50 EMA for intermediate trend
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # Stronger trend requirement: both EMAs must be rising
        df['ema_50_rising'] = df['ema_50'] > df['ema_50'].shift(5)
        df['ema_200_rising'] = df['ema_200'] > df['ema_200'].shift(5)
        df['strong_uptrend'] = (df['close'] > df['ema_50']) & \
                                (df['ema_50'] > df['ema_200']) & \
                                df['ema_50_rising'] & \
                                df['ema_200_rising']
        
        # Original MACD
        macd, signal, hist = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # NEW: Wait for MACD to be in positive territory for a while
        df['macd_positive'] = df['macd'] > 0
        df['macd_positive_3bars'] = (
            df['macd_positive'] & 
            df['macd_positive'].shift(1) & 
            df['macd_positive'].shift(2)
        )
        
        # NEW: Look for MACD pullback (better entry)
        # MACD was rising, now pulling back, ready to resume
        df['macd_pullback'] = (
            (df['macd'] < df['macd'].shift(1)) &  # Pulling back
            (df['macd'].shift(1) > df['macd'].shift(2)) &  # Was rising
            (df['macd'] > df['macd_signal'])  # Still above signal
        )
        
        # Original: MACD cross up
        # Refined: MACD crosses up OR bounces from pullback
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        df['macd_resume'] = (
            df['macd_pullback'].shift(1) &  # Yesterday was pullback
            (df['macd'] > df['macd'].shift(1))  # Today resuming up
        )
        
        # NEW: Volume confirmation (institutions must agree)
        df['volume_avg_20'] = df['volume'].rolling(20).mean()
        df['volume_surge'] = df['volume'] > (df['volume_avg_20'] * 1.2)  # 20% above avg
        
        # NEW: RSI confirmation (not overbought)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_healthy'] = (df['rsi'] > 50) & (df['rsi'] < 70)  # Bullish but not overbought
        
        # NEW: Price action confirmation (bullish candle)
        df['bullish_candle'] = df['close'] > df['open']
        
        # Entry: ALL conditions must be met
        df['signal'] = 0
        df.loc[
            df['strong_uptrend'] &  # Stronger trend requirement
            df['macd_positive_3bars'] &  # MACD established positive
            (df['macd_cross_up'] | df['macd_resume']) &  # Cross or resume
            df['volume_surge'] &  # Volume confirmation
            df['rsi_healthy'] &  # RSI confirmation
            df['bullish_candle'],  # Price action confirmation
            'signal'
        ] = 1
        
        # Exit: More aggressive (preserve profits)
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # NEW: Exit if trend weakens OR RSI overbought
        df['exit_signal'] = 0
        df.loc[
            df['macd_cross_down'] | 
            ~df['strong_uptrend'] |
            (df['rsi'] > 75),  # Exit if overbought
            'exit_signal'
        ] = 1
        
        return df


class MACDWithStopHunt:
    """
    MACD + Stop Hunt Detection
    Enter after stop hunt clears out weak hands
    """
    
    def __init__(self):
        self.name = "MACD Stop Hunt"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Trend
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['uptrend'] = df['close'] > df['ema_200']
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'], 12, 26, 9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        
        # Detect stop hunt (wick below support then recovery)
        df['support'] = df['low'].rolling(20).min()
        df['stop_hunt'] = (
            (df['low'] < df['support'].shift(1)) &  # Broke below support
            (df['close'] > df['open']) &  # But closed bullish
            (df['close'] > df['support'].shift(1))  # And recovered
        )
        
        # Volume on stop hunt
        df['volume_avg'] = df['volume'].rolling(20).mean()
        df['high_volume'] = df['volume'] > df['volume_avg'] * 1.5
        
        # Entry: After stop hunt clears + MACD bullish
        df['signal'] = 0
        df.loc[
            df['stop_hunt'] &
            df['macd_bullish'] &
            df['uptrend'] &
            df['high_volume'],
            'signal'
        ] = 1
        
        # Exit
        df['exit_signal'] = 0
        df.loc[
            ~df['macd_bullish'] | 
            ~df['uptrend'],
            'exit_signal'
        ] = 1
        
        return df


class MACDWithKeltnerChannels:
    """
    MACD + Keltner Channels
    Enter on squeeze breakout with MACD confirmation
    """
    
    def __init__(self):
        self.name = "MACD Keltner Squeeze"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'], 12, 26, 9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_positive'] = (df['macd'] > 0) & (df['macd'] > df['macd_signal'])
        
        # Keltner Channels
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
        df['keltner_upper'] = df['ema_20'] + (df['atr'] * 2)
        df['keltner_lower'] = df['ema_20'] - (df['atr'] * 2)
        
        # Bollinger Bands (for squeeze detection)
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_lower'] = lower
        
        # Squeeze: Bollinger inside Keltner
        df['squeeze'] = (
            (df['bb_upper'] < df['keltner_upper']) &
            (df['bb_lower'] > df['keltner_lower'])
        )
        
        # Squeeze release: Bollinger breaks out
        df['squeeze_release'] = (
            df['squeeze'].shift(1) &  # Was in squeeze
            ~df['squeeze'] &  # No longer in squeeze
            (df['close'] > df['keltner_upper'])  # Broke upper channel
        )
        
        # Trend
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['uptrend'] = df['close'] > df['ema_50']
        
        # Entry: Squeeze release + MACD positive + trend
        df['signal'] = 0
        df.loc[
            df['squeeze_release'] &
            df['macd_positive'] &
            df['uptrend'],
            'signal'
        ] = 1
        
        # Exit: Back below Keltner or MACD turns
        df['exit_signal'] = 0
        df.loc[
            (df['close'] < df['keltner_lower']) |
            ~df['macd_positive'],
            'exit_signal'
        ] = 1
        
        return df


class MACDDivergence:
    """
    MACD + Divergence Detection
    Hidden bullish divergence = strong trend continuation
    """
    
    def __init__(self):
        self.name = "MACD Divergence"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'], 12, 26, 9)
        df['macd'] = macd
        df['macd_signal'] = signal
        
        # Find swing lows in price and MACD
        df['price_low'] = df['low'].rolling(10, center=True).min()
        df['is_price_low'] = df['low'] == df['price_low']
        
        df['macd_low'] = df['macd'].rolling(10, center=True).min()
        df['is_macd_low'] = df['macd'] == df['macd_low']
        
        # Hidden bullish divergence:
        # Price makes higher low, MACD makes lower low
        # (Shows underlying strength despite pullback)
        df['price_low_value'] = np.where(df['is_price_low'], df['low'], np.nan)
        df['price_low_value'] = df['price_low_value'].ffill()
        
        df['macd_low_value'] = np.where(df['is_macd_low'], df['macd'], np.nan)
        df['macd_low_value'] = df['macd_low_value'].ffill()
        
        # Detect divergence
        df['hidden_bull_div'] = (
            (df['low'] > df['price_low_value'].shift(10)) &  # Higher price low
            (df['macd'] < df['macd_low_value'].shift(10)) &  # Lower MACD low
            df['is_macd_low']
        )
        
        # Trend context
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['uptrend'] = df['close'] > df['ema_50']
        
        # MACD must turn bullish
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        # Entry
        df['signal'] = 0
        df.loc[
            df['hidden_bull_div'] &
            df['uptrend'] &
            df['macd_cross_up'],
            'signal'
        ] = 1
        
        # Exit
        df['exit_signal'] = 0
        df.loc[
            (df['macd'] < df['macd_signal']) |
            ~df['uptrend'],
            'exit_signal'
        ] = 1
        
        return df


class AggressiveMACDScalp:
    """
    MACD Scalping - Quick in/out
    Target: Higher frequency, tight stops, quick profits
    """
    
    def __init__(self):
        self.name = "MACD Scalper"
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Fast MACD for scalping
        macd, signal, hist = talib.MACD(df['close'], 5, 13, 5)  # Faster settings
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # MACD histogram acceleration
        df['hist_accel'] = df['macd_hist'] > df['macd_hist'].shift(1)
        df['hist_positive'] = df['macd_hist'] > 0
        
        # Micro trend (20 EMA)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['above_ema'] = df['close'] > df['ema_20']
        
        # Volume burst
        df['volume_avg'] = df['volume'].rolling(10).mean()
        df['volume_spike'] = df['volume'] > df['volume_avg'] * 1.5
        
        # Entry: Fast MACD turn + volume + above EMA
        df['signal'] = 0
        df.loc[
            df['hist_accel'] &
            df['hist_positive'] &
            df['above_ema'] &
            df['volume_spike'],
            'signal'
        ] = 1
        
        # Exit: Quick - histogram turns or below EMA
        df['exit_signal'] = 0
        df.loc[
            ~df['hist_accel'] |
            (df['macd_hist'] < 0) |
            ~df['above_ema'],
            'exit_signal'
        ] = 1
        
        return df


def get_refined_macd_strategies():
    """Get all refined MACD variations"""
    return [
        RefinedMACDTrend(),
        MACDWithStopHunt(),
        MACDWithKeltnerChannels(),
        MACDDivergence(),
        AggressiveMACDScalp()
    ]


if __name__ == "__main__":
    print("🎯 Refined MACD Strategies")
    print("=" * 80)
    print("\nOriginal MACD: -0.23%, 35.9% WR, PF 1.09 (Best ML strategy)")
    print("\nGoal: Push into profitability with refinements")
    print("\n" + "=" * 80)
    
    for i, strategy in enumerate(get_refined_macd_strategies(), 1):
        print(f"{i}. {strategy.name}")
    print("=" * 80)
