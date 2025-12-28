"""
Backtesting Engine

This module provides a complete backtesting framework for testing trading strategies
on historical data. It simulates trades, calculates P&L, and generates performance metrics.

Key Features:
- Row-by-row historical replay
- Pluggable strategy system
- Transaction cost modeling (fees + slippage)
- Position tracking and P&L calculation
- Detailed trade logging
- Performance metrics (win rate, Sharpe ratio, max drawdown, etc.)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from config import (
    INITIAL_CAPITAL,
    POSITION_SIZE_PERCENT,
    MAKER_FEE,
    TAKER_FEE,
    SLIPPAGE,
    DEFAULT_STRATEGY_PARAMS,
)
from indicators import add_all_indicators

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PositionSide(Enum):
    """Position side"""
    LONG = "LONG"
    FLAT = "FLAT"


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    fees: float
    signal_type: str
    hold_time_hours: float


@dataclass
class Position:
    """Represents an open position"""
    entry_time: datetime
    entry_price: float
    quantity: float
    side: PositionSide
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


# =============================================================================
# STRATEGY BASE CLASS
# =============================================================================

class Strategy:
    """
    Base class for trading strategies.
    
    To create a custom strategy, inherit from this class and implement
    the generate_signal() method.
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize strategy with parameters.
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> SignalType:
        """
        Generate trading signal for current bar.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            index: Current bar index
        
        Returns:
            SignalType (BUY, SELL, or HOLD)
        """
        raise NotImplementedError("Subclasses must implement generate_signal()")
    
    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
        
        Returns:
            Stop loss price
        """
        multiplier = self.params.get('atr_multiplier', 1.5)
        return entry_price - (atr * multiplier)
    
    def calculate_take_profit(self, entry_price: float, atr: float) -> float:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
        
        Returns:
            Take profit price
        """
        multiplier = self.params.get('atr_multiplier', 1.5)
        return entry_price + (atr * multiplier * 2)  # 2:1 reward/risk


# =============================================================================
# DEFAULT EMA CROSSOVER STRATEGY
# =============================================================================

class EMACrossoverStrategy(Strategy):
    """
    EMA Crossover Strategy with ATR Filter.
    
    Entry Rules:
    - BUY when Fast EMA crosses above Slow EMA
    - ATR must be above minimum threshold (volatility filter)
    - RSI should not be overbought (> 70)
    
    Exit Rules:
    - SELL when Fast EMA crosses below Slow EMA
    - Or hit stop loss / take profit
    
    Default Parameters:
    - fast_ema: 20
    - slow_ema: 50
    - atr_period: 14
    - min_atr: 0.001
    - atr_multiplier: 1.5
    """
    
    def __init__(self, params: Dict = None):
        if params is None:
            params = DEFAULT_STRATEGY_PARAMS.copy()
        super().__init__(params)
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> SignalType:
        """
        Generate signal based on EMA crossover and ATR filter.
        
        Args:
            df: DataFrame with indicators
            index: Current bar index
        
        Returns:
            SignalType
        """
        # Need at least 2 bars for crossover detection
        if index < 1:
            return SignalType.HOLD
        
        # Current and previous values
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        
        # Check if indicators are ready
        if pd.isna(current['ema_fast']) or pd.isna(current['ema_slow']):
            return SignalType.HOLD
        
        # Get indicator values
        fast_ema_curr = current['ema_fast']
        slow_ema_curr = current['ema_slow']
        fast_ema_prev = previous['ema_fast']
        slow_ema_prev = previous['ema_slow']
        
        atr = current['atr']
        rsi = current.get('rsi', 50)  # Default to neutral if not available
        
        # ATR filter: volatility must be above minimum
        min_atr = self.params.get('min_atr', 0.001)
        if pd.isna(atr) or atr < min_atr:
            return SignalType.HOLD
        
        # Detect bullish crossover
        if fast_ema_curr > slow_ema_curr and fast_ema_prev <= slow_ema_prev:
            # Additional filter: not overbought
            if rsi < 70:
                return SignalType.BUY
        
        # Detect bearish crossover
        if fast_ema_curr < slow_ema_curr and fast_ema_prev >= slow_ema_prev:
            return SignalType.SELL
        
        return SignalType.HOLD


# =============================================================================
# BACKTESTER CLASS
# =============================================================================

class Backtester:
    """
    Main backtesting engine.
    
    This class handles:
    - Loading and preparing historical data
    - Running strategy signals row-by-row
    - Position management
    - Trade execution simulation
    - P&L calculation
    - Performance metrics
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = INITIAL_CAPITAL,
        position_size_pct: float = POSITION_SIZE_PERCENT,
        maker_fee: float = MAKER_FEE,
        taker_fee: float = TAKER_FEE,
        slippage: float = SLIPPAGE
    ):
        """
        Initialize backtester.
        
        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy instance
            initial_capital: Starting capital in USDT
            position_size_pct: Position size as % of capital
            maker_fee: Maker fee (as decimal, e.g., 0.001 = 0.1%)
            taker_fee: Taker fee
            slippage: Slippage assumption (as decimal)
        """
        self.df = df.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        
        # State tracking
        self.capital = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        
        logger.info(f"Backtester initialized with ${initial_capital:,.2f} capital")
        logger.info(f"Strategy: {strategy.name}")
    
    def prepare_data(self):
        """
        Prepare data by adding all necessary indicators.
        """
        logger.info("Preparing data and computing indicators...")
        self.df = add_all_indicators(self.df, self.strategy.params)
        logger.info(f"Data prepared: {len(self.df)} bars")
    
    def execute_trade(
        self,
        price: float,
        quantity: float,
        side: str,
        timestamp: datetime
    ) -> float:
        """
        Execute a trade and return fees paid.
        
        Args:
            price: Execution price
            quantity: Quantity to trade
            side: 'BUY' or 'SELL'
            timestamp: Trade timestamp
        
        Returns:
            Total fees paid
        """
        # Apply slippage
        if side == 'BUY':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate notional value
        notional = execution_price * quantity
        
        # Calculate fees (using taker fee for simplicity)
        fees = notional * self.taker_fee
        
        logger.debug(f"{side} {quantity:.6f} @ {execution_price:.2f} (fees: ${fees:.2f})")
        
        return fees
    
    def open_position(self, index: int):
        """
        Open a long position.
        
        Args:
            index: Current bar index
        """
        if self.position is not None:
            return  # Already in position
        
        current = self.df.iloc[index]
        entry_price = current['close']
        timestamp = current['open_time']
        
        # Calculate position size
        position_value = self.capital * self.position_size_pct
        quantity = position_value / entry_price
        
        # Execute trade
        fees = self.execute_trade(entry_price, quantity, 'BUY', timestamp)
        
        # Update capital (subtract position value and fees)
        self.capital -= (position_value + fees)
        
        # Calculate stop loss and take profit
        atr = current['atr']
        stop_loss = self.strategy.calculate_stop_loss(entry_price, atr)
        take_profit = self.strategy.calculate_take_profit(entry_price, atr)
        
        # Create position
        self.position = Position(
            entry_time=timestamp,
            entry_price=entry_price,
            quantity=quantity,
            side=PositionSide.LONG,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        logger.info(f"OPENED LONG @ ${entry_price:.2f} | Qty: {quantity:.6f} | "
                   f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
    
    def close_position(self, index: int, reason: str = "SIGNAL"):
        """
        Close current position.
        
        Args:
            index: Current bar index
            reason: Reason for closing (SIGNAL, STOP_LOSS, TAKE_PROFIT)
        """
        if self.position is None:
            return  # No position to close
        
        current = self.df.iloc[index]
        exit_price = current['close']
        timestamp = current['open_time']
        
        # Execute trade
        fees = self.execute_trade(
            exit_price,
            self.position.quantity,
            'SELL',
            timestamp
        )
        
        # Calculate P&L
        gross_pnl = (exit_price - self.position.entry_price) * self.position.quantity
        net_pnl = gross_pnl - fees
        pnl_percent = (net_pnl / (self.position.entry_price * self.position.quantity)) * 100
        
        # Update capital
        self.capital += (exit_price * self.position.quantity - fees)
        
        # Calculate hold time
        hold_time = timestamp - self.position.entry_time
        hold_time_hours = hold_time.total_seconds() / 3600
        
        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            entry_price=self.position.entry_price,
            exit_time=timestamp,
            exit_price=exit_price,
            quantity=self.position.quantity,
            pnl=net_pnl,
            pnl_percent=pnl_percent,
            fees=fees,
            signal_type=reason,
            hold_time_hours=hold_time_hours
        )
        
        self.trades.append(trade)
        
        logger.info(f"CLOSED {reason} @ ${exit_price:.2f} | "
                   f"P&L: ${net_pnl:+.2f} ({pnl_percent:+.2f}%) | "
                   f"Hold: {hold_time_hours:.1f}h")
        
        # Clear position
        self.position = None
    
    def check_stop_loss_take_profit(self, index: int) -> bool:
        """
        Check if stop loss or take profit is hit.
        
        Args:
            index: Current bar index
        
        Returns:
            True if position was closed, False otherwise
        """
        if self.position is None:
            return False
        
        current = self.df.iloc[index]
        low = current['low']
        high = current['high']
        
        # Check stop loss
        if self.position.stop_loss and low <= self.position.stop_loss:
            self.close_position(index, reason="STOP_LOSS")
            return True
        
        # Check take profit
        if self.position.take_profit and high >= self.position.take_profit:
            self.close_position(index, reason="TAKE_PROFIT")
            return True
        
        return False
    
    def run(self) -> pd.DataFrame:
        """
        Run the backtest.
        
        This is the main method that:
        1. Prepares data (adds indicators)
        2. Iterates through each bar
        3. Generates signals
        4. Manages positions
        5. Records equity curve
        
        Returns:
            DataFrame with trade results
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)
        
        # Prepare data
        self.prepare_data()
        
        # Iterate through each bar
        for index in range(len(self.df)):
            current = self.df.iloc[index]
            
            # Track equity
            equity = self.capital
            if self.position:
                equity += current['close'] * self.position.quantity
            self.equity_curve.append(equity)
            
            # Check stop loss / take profit first
            if self.check_stop_loss_take_profit(index):
                continue
            
            # Generate signal
            signal = self.strategy.generate_signal(self.df, index)
            
            # Execute based on signal
            if signal == SignalType.BUY and self.position is None:
                self.open_position(index)
            
            elif signal == SignalType.SELL and self.position is not None:
                self.close_position(index, reason="SIGNAL")
        
        # Close any open position at end
        if self.position is not None:
            logger.info("Closing position at end of backtest")
            self.close_position(len(self.df) - 1, reason="END")
        
        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)
        
        # Return results
        return self.get_results()
    
    def get_results(self) -> pd.DataFrame:
        """
        Get backtest results as DataFrame.
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            logger.warning("No trades executed")
            return pd.DataFrame()
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        
        return trades_df
    
    def print_summary(self):
        """
        Print backtest performance summary.
        """
        if not self.trades:
            print("\nNo trades executed")
            return
        
        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl < 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_fees = sum(t.pnl for t in self.trades)
        
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        final_capital = self.capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Print summary
        print("\n" + "=" * 80)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"\nStrategy: {self.strategy.name}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total P&L: ${total_pnl:+,.2f}")
        print(f"Total Fees: ${total_fees:,.2f}")
        
        print(f"\nTrades: {total_trades}")
        print(f"Winning: {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing: {losing_trades} ({100-win_rate:.1f}%)")
        
        print(f"\nAverage Win: ${avg_win:+,.2f}")
        print(f"Average Loss: ${avg_loss:+,.2f}")
        if avg_loss != 0:
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades))
            print(f"Profit Factor: {profit_factor:.2f}")
        
        print(f"\nMax Drawdown: {max_drawdown:.2f}%")
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            if np.std(returns) > 0:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
                print(f"Sharpe Ratio: {sharpe:.2f}")
        
        print("=" * 80)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example backtesting workflow.
    
    Run this file directly to test the backtester:
        python backtester.py
    """
    from datetime import timedelta
    from data_fetcher import get_historical_data
    
    print("=" * 80)
    print("BACKTESTER EXAMPLE")
    print("=" * 80)
    
    # 1. Fetch historical data
    print("\n1. Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    df = get_historical_data("BTCUSDT", "1h", start_date, end_date)
    print(f"   Loaded {len(df)} candles")
    
    # 2. Create strategy
    print("\n2. Initializing strategy...")
    strategy = EMACrossoverStrategy()
    print(f"   Strategy: {strategy.name}")
    print(f"   Parameters: {strategy.params}")
    
    # 3. Run backtest
    print("\n3. Running backtest...")
    backtester = Backtester(df, strategy)
    results = backtester.run()
    
    # 4. Print summary
    backtester.print_summary()
    
    # 5. Show sample trades
    if not results.empty:
        print("\n" + "=" * 80)
        print("SAMPLE TRADES (First 5)")
        print("=" * 80)
        print(results[['entry_time', 'entry_price', 'exit_time', 'exit_price', 
                      'pnl', 'pnl_percent']].head().to_string())
