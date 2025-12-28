"""
Deterministic Trading Strategies - No ML Required
Pure technical analysis with strict, testable rules
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import talib


class DeterministicStrategy:
    """Base class for deterministic strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = []
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals - to be implemented by subclasses"""
        raise NotImplementedError
        
    def calculate_position_size(self, capital: float, risk_per_trade: float = 0.01) -> float:
        """Conservative position sizing - never risk more than 1% per trade"""
        return capital * risk_per_trade


class TripleConfirmationStrategy(DeterministicStrategy):
    """
    Strategy: Only enter when 3 indicators align
    - EMA crossover (trend)
    - RSI confirmation (momentum)
    - Volume surge (strength)
    
    Rules:
    BUY: Fast EMA > Slow EMA AND RSI > 50 AND Volume > 1.5x avg
    SELL: Fast EMA < Slow EMA OR RSI < 30 OR stop loss hit
    
    Conservative: Waits for strong confirmation, exits quickly on weakness
    """
    
    def __init__(self):
        super().__init__("Triple Confirmation")
        self.fast_period = 12
        self.slow_period = 26
        self.rsi_period = 14
        self.volume_multiplier = 1.5
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with triple confirmation"""
        df = df.copy()
        
        # Calculate indicators
        df['ema_fast'] = talib.EMA(df['close'], timeperiod=self.fast_period)
        df['ema_slow'] = talib.EMA(df['close'], timeperiod=self.slow_period)
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        df['volume_avg'] = df['volume'].rolling(20).mean()
        
        # Trend confirmation
        df['trend_bullish'] = df['ema_fast'] > df['ema_slow']
        
        # Momentum confirmation
        df['momentum_bullish'] = df['rsi'] > 50
        df['momentum_strong'] = df['rsi'] > 60  # Stronger signal
        
        # Volume confirmation
        df['volume_surge'] = df['volume'] > (df['volume_avg'] * self.volume_multiplier)
        
        # Triple confirmation for entry
        df['signal'] = 0
        df.loc[
            df['trend_bullish'] & 
            df['momentum_strong'] & 
            df['volume_surge'],
            'signal'
        ] = 1
        
        # Exit signals
        df['exit_signal'] = 0
        df.loc[
            ~df['trend_bullish'] | 
            (df['rsi'] < 30),
            'exit_signal'
        ] = 1
        
        return df


class BreakoutRetestStrategy(DeterministicStrategy):
    """
    Strategy: Trade breakouts with retest confirmation
    - Identify strong support/resistance levels
    - Wait for breakout
    - Enter on retest of breakout level
    
    Rules:
    BUY: Price breaks above resistance, retests, holds, and bounces
    SELL: Price breaks below support or fails retest
    
    Conservative: Waits for confirmation, avoids false breakouts
    """
    
    def __init__(self):
        super().__init__("Breakout Retest")
        self.lookback = 20
        self.retest_threshold = 0.005  # 0.5% tolerance for retest
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout + retest signals"""
        df = df.copy()
        
        # Identify resistance levels (rolling highs)
        df['resistance'] = df['high'].rolling(self.lookback).max()
        df['support'] = df['low'].rolling(self.lookback).min()
        
        # Detect breakouts
        df['breakout_up'] = (df['close'] > df['resistance'].shift(1)) & \
                            (df['close'].shift(1) <= df['resistance'].shift(2))
        
        # Mark breakout levels
        df['breakout_level'] = np.nan
        df.loc[df['breakout_up'], 'breakout_level'] = df['resistance'].shift(1)
        df['breakout_level'] = df['breakout_level'].ffill()
        
        # Detect retest (price comes back to within 0.5% of breakout level)
        df['near_breakout'] = np.abs(df['low'] - df['breakout_level']) / df['breakout_level'] < self.retest_threshold
        
        # Detect bounce from retest
        df['bounce'] = (df['close'] > df['open']) & df['near_breakout']
        
        # Signal: Buy on bounce after retest
        df['signal'] = 0
        df.loc[df['bounce'], 'signal'] = 1
        
        # Exit: Break below support or failed retest
        df['exit_signal'] = 0
        df.loc[df['close'] < df['support'], 'exit_signal'] = 1
        
        return df


class TrendFollowingMACD(DeterministicStrategy):
    """
    Strategy: Pure trend following with MACD
    - Only trade with the trend
    - Use MACD for timing
    - Strict stop losses
    
    Rules:
    BUY: Price > 200 EMA (uptrend) AND MACD crosses above signal AND MACD > 0
    SELL: MACD crosses below signal OR price < 200 EMA
    
    Conservative: Only trades strong trends, exits at first sign of reversal
    """
    
    def __init__(self):
        super().__init__("Trend Following MACD")
        self.trend_period = 200
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-following signals"""
        df = df.copy()
        
        # Long-term trend
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
        df['macd_hist'] = hist
        
        # MACD crossover
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & \
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & \
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Entry: Uptrend + MACD bullish cross + MACD above zero
        df['signal'] = 0
        df.loc[
            df['in_uptrend'] & 
            df['macd_cross_up'] & 
            (df['macd'] > 0),
            'signal'
        ] = 1
        
        # Exit: MACD bearish cross or trend break
        df['exit_signal'] = 0
        df.loc[
            df['macd_cross_down'] | 
            ~df['in_uptrend'],
            'exit_signal'
        ] = 1
        
        return df


class BollingerBandMeanReversion(DeterministicStrategy):
    """
    Strategy: Mean reversion with Bollinger Bands
    - Buy oversold bounces
    - Sell overbought rejections
    - Only in ranging markets
    
    Rules:
    BUY: Price touches lower band AND RSI < 30 AND closes back inside bands
    SELL: Price reaches middle band OR RSI > 70
    
    Conservative: Small positions, quick exits, only in low volatility
    """
    
    def __init__(self):
        super().__init__("Bollinger Mean Reversion")
        self.bb_period = 20
        self.bb_std = 2
        self.rsi_period = 14
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        df = df.copy()
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # Detect ranging market (low ADX)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['ranging'] = df['adx'] < 25
        
        # Touch lower band
        df['touched_lower'] = df['low'] <= df['bb_lower']
        
        # Close back inside
        df['back_inside'] = (df['close'] > df['bb_lower']) & df['touched_lower'].shift(1)
        
        # Entry: Oversold bounce in ranging market
        df['signal'] = 0
        df.loc[
            df['ranging'] &
            df['back_inside'] & 
            (df['rsi'] < 30),
            'signal'
        ] = 1
        
        # Exit: Reach middle band or overbought
        df['exit_signal'] = 0
        df.loc[
            (df['close'] >= df['bb_middle']) | 
            (df['rsi'] > 70),
            'exit_signal'
        ] = 1
        
        return df


class MomentumScalper(DeterministicStrategy):
    """
    Strategy: Quick momentum scalps
    - Catch strong momentum moves
    - Very tight stops
    - Quick profits
    
    Rules:
    BUY: 3 consecutive green candles AND RSI > 60 AND increasing volume
    SELL: 1 red candle OR 0.5% profit OR 0.3% loss
    
    Conservative: Very short holding period, tight risk management
    """
    
    def __init__(self):
        super().__init__("Momentum Scalper")
        self.profit_target = 0.005  # 0.5%
        self.stop_loss = 0.003  # 0.3%
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate scalping signals"""
        df = df.copy()
        
        # Consecutive green candles
        df['green_candle'] = df['close'] > df['open']
        df['consec_green'] = (
            df['green_candle'] & 
            df['green_candle'].shift(1) & 
            df['green_candle'].shift(2)
        )
        
        # RSI momentum
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['strong_momentum'] = df['rsi'] > 60
        
        # Increasing volume
        df['volume_increasing'] = (
            (df['volume'] > df['volume'].shift(1)) &
            (df['volume'].shift(1) > df['volume'].shift(2))
        )
        
        # Entry signal
        df['signal'] = 0
        df.loc[
            df['consec_green'] & 
            df['strong_momentum'] & 
            df['volume_increasing'],
            'signal'
        ] = 1
        
        # Exit on any red candle
        df['exit_signal'] = 0
        df.loc[~df['green_candle'], 'exit_signal'] = 1
        
        return df


class SmartMoneyConceptsSMC(DeterministicStrategy):
    """
    Strategy: Smart Money Concepts (Order Blocks + FVG)
    - Identify institutional order blocks
    - Trade fair value gaps
    - Follow smart money flow
    
    Rules:
    BUY: Price returns to bullish order block + creates FVG + breaks structure
    SELL: Opposite order block touched or liquidity sweep
    
    Conservative: Waits for institutional confirmation, respects key levels
    """
    
    def __init__(self):
        super().__init__("Smart Money Concepts")
        self.swing_lookback = 10
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate SMC-based signals"""
        df = df.copy()
        
        # Identify swing highs/lows
        df['swing_high'] = df['high'] == df['high'].rolling(self.swing_lookback, center=True).max()
        df['swing_low'] = df['low'] == df['low'].rolling(self.swing_lookback, center=True).min()
        
        # Break of structure (BOS)
        df['prev_swing_high'] = df.loc[df['swing_high'], 'high'].ffill()
        df['bos'] = df['close'] > df['prev_swing_high']
        
        # Fair Value Gap (FVG) - gap between candles
        df['fvg_up'] = df['low'] > df['high'].shift(2)
        
        # Order block (last down candle before strong up move)
        df['strong_up_move'] = (df['close'] > df['open']) & \
                                ((df['close'] - df['open']) > 2 * (df['open'] - df['close']).shift(1))
        df['order_block'] = (df['close'] < df['open']) & df['strong_up_move'].shift(-1)
        
        # Mark order block levels
        df['ob_level'] = np.nan
        df.loc[df['order_block'], 'ob_level'] = df['low']
        df['ob_level'] = df['ob_level'].ffill()
        
        # Price returns to order block
        df['at_ob'] = (df['low'] <= df['ob_level'] * 1.01) & (df['low'] >= df['ob_level'] * 0.99)
        
        # Entry: Price at order block + FVG + BOS
        df['signal'] = 0
        df.loc[
            df['at_ob'] & 
            df['fvg_up'] & 
            df['bos'],
            'signal'
        ] = 1
        
        # Exit: Opposite swing or liquidity sweep
        df['exit_signal'] = 0
        df.loc[df['swing_high'], 'exit_signal'] = 1
        
        return df


def get_all_strategies() -> List[DeterministicStrategy]:
    """Get all available deterministic strategies"""
    return [
        TripleConfirmationStrategy(),
        BreakoutRetestStrategy(),
        TrendFollowingMACD(),
        BollingerBandMeanReversion(),
        MomentumScalper(),
        SmartMoneyConceptsSMC()
    ]


def print_strategy_info():
    """Print information about all strategies"""
    strategies = get_all_strategies()
    
    print("=" * 80)
    print("DETERMINISTIC TRADING STRATEGIES")
    print("=" * 80)
    print()
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy.name}")
        print(f"   {strategy.__doc__.strip()}")
        print()
    
    print("=" * 80)
    print("All strategies include:")
    print("  • Conservative position sizing (1% risk per trade)")
    print("  • Strict entry rules (no guessing)")
    print("  • Clear exit signals (no emotions)")
    print("  • Capital preservation focus")
    print("=" * 80)


if __name__ == "__main__":
    print_strategy_info()
