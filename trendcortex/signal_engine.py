"""
Signal Engine Module

Generates trading signals based on technical indicators and rule-based strategies.
Combines multiple indicators to produce high-confidence trading signals.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from trendcortex.config import Config
from trendcortex.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    apply_all_indicators,
)
from trendcortex.logger import get_logger


class SignalType(Enum):
    """Types of trading signals"""
    EMA_CROSS = "ema_cross"
    RSI_EXTREME = "rsi_extreme"
    BB_BREAKOUT = "bb_breakout"
    MACD_CROSS = "macd_cross"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    VOLUME_SPIKE = "volume_spike"
    MULTI_INDICATOR = "multi_indicator"


class SignalDirection(Enum):
    """Signal direction"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    price: float
    indicators: Dict[str, float]
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict] = None


class SignalEngine:
    """
    Generates trading signals based on technical analysis.
    
    Implements multiple strategies:
    - EMA crossover
    - RSI extreme levels
    - Bollinger Band breakouts
    - MACD crossovers
    - Trend following with confirmation
    - Mean reversion
    - Volume-based signals
    """
    
    def __init__(self, config: Config):
        """
        Initialize signal engine.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = get_logger()
        
        # Signal configuration
        self.enable_long = config.signals.enable_long
        self.enable_short = config.signals.enable_short
        self.min_confidence = config.signals.min_confidence
        self.require_trend_alignment = config.signals.require_trend_alignment
        self.require_volume_confirmation = config.signals.require_volume_confirmation
        
        # Indicator parameters
        self.ema_fast = config.indicators.ema["fast_period"]
        self.ema_slow = config.indicators.ema["slow_period"]
        self.ema_trend = config.indicators.ema["trend_period"]
        self.rsi_period = config.indicators.rsi["period"]
        self.rsi_overbought = config.indicators.rsi["overbought"]
        self.rsi_oversold = config.indicators.rsi["oversold"]
        self.atr_period = config.indicators.atr["period"]
        self.atr_multiplier = config.indicators.atr["multiplier"]
    
    def generate_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> List[TradingSignal]:
        """
        Generate trading signals from market data.
        
        Args:
            symbol: Trading pair symbol
            df: DataFrame with OHLCV data
            
        Returns:
            List of trading signals
        """
        if df.empty or len(df) < max(self.ema_trend, 50):
            self.logger.warning(f"Insufficient data for signal generation: {len(df)} candles")
            return []
        
        # Apply all indicators
        df = apply_all_indicators(df, self.config.indicators.dict())
        
        signals = []
        
        # Generate signals from different strategies
        if self.enable_long:
            # Long signals
            ema_cross_long = self._check_ema_crossover(df, "long")
            if ema_cross_long:
                signals.append(ema_cross_long)
            
            rsi_oversold = self._check_rsi_extreme(df, "long")
            if rsi_oversold:
                signals.append(rsi_oversold)
            
            bb_lower = self._check_bollinger_breakout(df, "long")
            if bb_lower:
                signals.append(bb_lower)
            
            macd_cross_long = self._check_macd_cross(df, "long")
            if macd_cross_long:
                signals.append(macd_cross_long)
        
        if self.enable_short:
            # Short signals
            ema_cross_short = self._check_ema_crossover(df, "short")
            if ema_cross_short:
                signals.append(ema_cross_short)
            
            rsi_overbought = self._check_rsi_extreme(df, "short")
            if rsi_overbought:
                signals.append(rsi_overbought)
            
            bb_upper = self._check_bollinger_breakout(df, "short")
            if bb_upper:
                signals.append(bb_upper)
            
            macd_cross_short = self._check_macd_cross(df, "short")
            if macd_cross_short:
                signals.append(macd_cross_short)
        
        # Filter by confidence threshold
        signals = [s for s in signals if s.confidence >= self.min_confidence]
        
        # Add multi-indicator confirmation signal if multiple signals align
        if len(signals) >= 2:
            multi_signal = self._create_multi_indicator_signal(df, signals)
            if multi_signal:
                signals.append(multi_signal)
        
        return signals
    
    def _check_ema_crossover(self, df: pd.DataFrame, direction: str) -> Optional[TradingSignal]:
        """
        Check for EMA crossover signal.
        
        Args:
            df: DataFrame with indicators
            direction: "long" or "short"
            
        Returns:
            Signal if crossover detected
        """
        if len(df) < 3:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check for valid EMA values
        if pd.isna(current["ema_fast"]) or pd.isna(current["ema_slow"]):
            return None
        
        if direction == "long":
            # Bullish crossover: fast crosses above slow
            if (previous["ema_fast"] <= previous["ema_slow"] and 
                current["ema_fast"] > current["ema_slow"]):
                
                # Trend confirmation
                if self.require_trend_alignment:
                    if current["close"] < current["ema_trend"]:
                        return None  # Against trend
                
                confidence = self._calculate_crossover_confidence(df, "long")
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.EMA_CROSS,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"crossover_type": "bullish"},
                )
        
        elif direction == "short":
            # Bearish crossover: fast crosses below slow
            if (previous["ema_fast"] >= previous["ema_slow"] and 
                current["ema_fast"] < current["ema_slow"]):
                
                # Trend confirmation
                if self.require_trend_alignment:
                    if current["close"] > current["ema_trend"]:
                        return None  # Against trend
                
                confidence = self._calculate_crossover_confidence(df, "short")
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.EMA_CROSS,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"crossover_type": "bearish"},
                )
        
        return None
    
    def _check_rsi_extreme(self, df: pd.DataFrame, direction: str) -> Optional[TradingSignal]:
        """
        Check for RSI extreme levels (oversold/overbought).
        
        Args:
            df: DataFrame with indicators
            direction: "long" or "short"
            
        Returns:
            Signal if RSI extreme detected
        """
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        if pd.isna(current["rsi"]):
            return None
        
        if direction == "long":
            # RSI oversold and starting to turn up
            if current["rsi"] < self.rsi_oversold and current["rsi"] > previous["rsi"]:
                confidence = (self.rsi_oversold - current["rsi"]) / self.rsi_oversold + 0.5
                confidence = min(confidence, 1.0)
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.RSI_EXTREME,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"rsi_value": current["rsi"]},
                )
        
        elif direction == "short":
            # RSI overbought and starting to turn down
            if current["rsi"] > self.rsi_overbought and current["rsi"] < previous["rsi"]:
                confidence = (current["rsi"] - self.rsi_overbought) / (100 - self.rsi_overbought) + 0.5
                confidence = min(confidence, 1.0)
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.RSI_EXTREME,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"rsi_value": current["rsi"]},
                )
        
        return None
    
    def _check_bollinger_breakout(self, df: pd.DataFrame, direction: str) -> Optional[TradingSignal]:
        """
        Check for Bollinger Band breakout/bounce.
        
        Args:
            df: DataFrame with indicators
            direction: "long" or "short"
            
        Returns:
            Signal if BB breakout detected
        """
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        if pd.isna(current["bb_upper"]) or pd.isna(current["bb_lower"]):
            return None
        
        if direction == "long":
            # Price bouncing off lower band
            if (previous["close"] <= previous["bb_lower"] and 
                current["close"] > current["bb_lower"]):
                
                # Calculate confidence based on BB position
                bb_position = (current["close"] - current["bb_lower"]) / (current["bb_upper"] - current["bb_lower"])
                confidence = 0.6 + (0.4 * (1 - bb_position))
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.BB_BREAKOUT,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"bb_position": bb_position},
                )
        
        elif direction == "short":
            # Price rejected from upper band
            if (previous["close"] >= previous["bb_upper"] and 
                current["close"] < current["bb_upper"]):
                
                bb_position = (current["close"] - current["bb_lower"]) / (current["bb_upper"] - current["bb_lower"])
                confidence = 0.6 + (0.4 * bb_position)
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.BB_BREAKOUT,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"bb_position": bb_position},
                )
        
        return None
    
    def _check_macd_cross(self, df: pd.DataFrame, direction: str) -> Optional[TradingSignal]:
        """
        Check for MACD crossover signal.
        
        Args:
            df: DataFrame with indicators
            direction: "long" or "short"
            
        Returns:
            Signal if MACD cross detected
        """
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        if pd.isna(current["macd"]) or pd.isna(current["macd_signal"]):
            return None
        
        if direction == "long":
            # Bullish MACD cross
            if (previous["macd"] <= previous["macd_signal"] and 
                current["macd"] > current["macd_signal"]):
                
                # Stronger signal if histogram is increasing
                hist_increasing = current["macd_hist"] > previous["macd_hist"]
                confidence = 0.7 if hist_increasing else 0.6
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.MACD_CROSS,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    metadata={"macd_histogram": current["macd_hist"]},
                )
        
        elif direction == "short":
            # Bearish MACD cross
            if (previous["macd"] >= previous["macd_signal"] and 
                current["macd"] < current["macd_signal"]):
                
                hist_decreasing = current["macd_hist"] < previous["macd_hist"]
                confidence = 0.7 if hist_decreasing else 0.6
                
                return self._create_signal(
                    df=df,
                    signal_type=SignalType.MACD_CROSS,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    metadata={"macd_histogram": current["macd_hist"]},
                )
        
        return None
    
    def _calculate_crossover_confidence(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate confidence score for crossover signals."""
        current = df.iloc[-1]
        
        # Base confidence
        confidence = 0.7
        
        # Increase if volume is elevated
        if self.require_volume_confirmation:
            if "volume_ratio" in current and current["volume_ratio"] > 1.2:
                confidence += 0.1
        
        # Increase if aligned with trend
        if direction == "long":
            if "ema_trend" in current and current["close"] > current["ema_trend"]:
                confidence += 0.1
        else:
            if "ema_trend" in current and current["close"] < current["ema_trend"]:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_multi_indicator_signal(
        self,
        df: pd.DataFrame,
        signals: List[TradingSignal],
    ) -> Optional[TradingSignal]:
        """
        Create a high-confidence signal when multiple indicators align.
        
        Args:
            df: DataFrame with indicators
            signals: List of individual signals
            
        Returns:
            Combined signal with higher confidence
        """
        # Check if all signals agree on direction
        directions = [s.direction for s in signals]
        if len(set(directions)) > 1:
            return None  # Conflicting signals
        
        direction = directions[0]
        
        # Average confidence weighted by signal type
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # Boost confidence for multiple confirmations
        boost = min(0.1 * (len(signals) - 1), 0.3)
        final_confidence = min(avg_confidence + boost, 1.0)
        
        # Combine metadata
        combined_metadata = {
            "num_confirmations": len(signals),
            "signal_types": [s.signal_type.value for s in signals],
        }
        
        return self._create_signal(
            df=df,
            signal_type=SignalType.MULTI_INDICATOR,
            direction=direction,
            confidence=final_confidence,
            metadata=combined_metadata,
        )
    
    def _create_signal(
        self,
        df: pd.DataFrame,
        signal_type: SignalType,
        direction: SignalDirection,
        confidence: float,
        metadata: Optional[Dict] = None,
    ) -> TradingSignal:
        """
        Create a trading signal with stop loss and take profit levels.
        
        Args:
            df: DataFrame with indicators
            signal_type: Type of signal
            direction: Signal direction
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            Complete trading signal
        """
        current = df.iloc[-1]
        
        # Entry price (current close)
        entry_price = current["close"]
        
        # Calculate stop loss and take profit using ATR
        atr = current.get("atr", entry_price * 0.02)  # Default 2% if ATR unavailable
        
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - (atr * self.atr_multiplier)
            take_profit = entry_price + (atr * self.atr_multiplier * 2)  # 2:1 R/R
        else:  # SHORT
            stop_loss = entry_price + (atr * self.atr_multiplier)
            take_profit = entry_price - (atr * self.atr_multiplier * 2)
        
        # Collect indicator values
        indicator_values = {
            "ema_fast": current.get("ema_fast", 0),
            "ema_slow": current.get("ema_slow", 0),
            "ema_trend": current.get("ema_trend", 0),
            "rsi": current.get("rsi", 0),
            "atr": current.get("atr", 0),
            "atr_percent": current.get("atr_percent", 0),
            "bb_position": current.get("bb_position", 0),
            "macd": current.get("macd", 0),
            "macd_signal": current.get("macd_signal", 0),
            "volume_ratio": current.get("volume_ratio", 0),
        }
        
        return TradingSignal(
            timestamp=current.get("timestamp", datetime.now()),
            symbol=self.config.trading.primary_symbol,
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            price=entry_price,
            indicators=indicator_values,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )
