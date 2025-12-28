"""
AI Strategy Backtester
======================

Integrates ML model predictions and LLM decision gating with the existing
backtesting engine to evaluate AI-driven trading strategies.

This backtester:
1. Trains ML models on historical data
2. Generates predictions for each candle
3. Uses LLM gate to approve/reject trades
4. Simulates trade execution with risk management
5. Tracks both ML metrics and trading performance

Key Features:
- Offline backtesting (no live API required)
- ML-aware metrics (prediction accuracy, confidence distribution)
- LLM decision tracking (approval rate, reasoning analysis)
- Traditional trading metrics (win rate, Sharpe, drawdown)
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

from ai_strategy.config import AIStrategyConfig
from ai_strategy.model_engine import ModelEngine
from ai_strategy.llm_gate import LLMGate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIBacktester:
    """
    AI-powered backtesting engine that combines ML predictions with LLM decision gating
    """
    
    def __init__(self, config: AIStrategyConfig):
        """
        Initialize the AI backtester
        
        Args:
            config: AI strategy configuration
        """
        self.config = config
        self.model_engine = ModelEngine(config)
        self.llm_gate = LLMGate(config)
        
        # Backtesting state
        self.trades: List[Dict] = []
        self.positions: List[Dict] = []
        self.equity_curve: List[float] = []
        self.ml_predictions: List[Dict] = []
        self.llm_decisions: List[Dict] = []
        
        # Performance metrics
        self.metrics: Dict = {}
        
        # Initial capital
        self.initial_capital = config.risk.initial_capital
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        
        logger.info("AI Backtester initialized")
    
    def prepare_data(self, df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets chronologically
        
        Args:
            df: Historical OHLCV data with indicators
            train_size: Proportion of data for training (0-1)
        
        Returns:
            (train_df, test_df): Training and testing DataFrames
        """
        split_idx = int(len(df) * train_size)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Data split: {len(train_df)} train, {len(test_df)} test candles")
        return train_df, test_df
    
    def train_models(self, train_df: pd.DataFrame) -> Dict:
        """
        Train ML models on training data
        
        Args:
            train_df: Training data
        
        Returns:
            Trained models dictionary
        """
        logger.info("Training ML models...")
        models = self.model_engine.train_models(train_df)
        
        # Log training metrics
        for name, metrics in self.model_engine.training_metrics.items():
            logger.info(f"{name}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
        
        return models
    
    def calculate_position_size(self, price: float, ml_confidence: float, atr: float) -> float:
        """
        Calculate position size based on ML confidence and volatility
        
        Args:
            price: Current price
            ml_confidence: ML prediction probability (0-1)
            atr: Average True Range
        
        Returns:
            Position size in base currency units
        """
        # Base position size (% of capital)
        base_size_pct = self.config.risk.max_position_size_percent
        
        # Adjust by ML confidence (50-100% of base size)
        confidence_multiplier = 0.5 + (ml_confidence * 0.5)
        adjusted_size_pct = base_size_pct * confidence_multiplier
        
        # Calculate size in currency
        size_in_currency = self.capital * adjusted_size_pct
        
        # Convert to units
        units = size_in_currency / price
        
        return units
    
    def calculate_stops(self, entry_price: float, direction: str, atr: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range
        
        Returns:
            (stop_loss, take_profit) prices
        """
        sl_distance = atr * self.config.risk.stop_loss_atr_multiplier
        tp_distance = atr * self.config.risk.take_profit_atr_multiplier
        
        if direction == 'long':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit
    
    def check_position_limits(self) -> bool:
        """
        Check if we can open a new position
        
        Returns:
            True if we can open a position, False otherwise
        """
        open_positions = [p for p in self.positions if p['status'] == 'open']
        return len(open_positions) < self.config.risk.max_open_positions
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        symbol: str,
        model_name: str = "random_forest",
        use_llm_gate: bool = True,
        train_size: float = 0.7
    ) -> Dict:
        """
        Run complete AI backtest
        
        Args:
            df: Historical OHLCV data with indicators
            symbol: Trading symbol
            model_name: ML model to use ('random_forest', 'xgboost', 'logistic_regression')
            use_llm_gate: Whether to use LLM decision gate
            train_size: Proportion of data for training
        
        Returns:
            Dictionary of backtest results and metrics
        """
        logger.info(f"Starting AI backtest for {symbol}")
        logger.info(f"Using model: {model_name}, LLM gate: {use_llm_gate}")
        
        # Split data
        train_df, test_df = self.prepare_data(df, train_size)
        
        # Train models
        models = self.train_models(train_df)
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
        
        # Generate predictions for test data
        logger.info(f"Generating predictions for {len(test_df)} test candles...")
        test_df_with_predictions = self.model_engine.predict_signals(test_df, model_name)
        
        # Reset state
        self.trades = []
        self.positions = []
        self.equity_curve = [self.initial_capital]
        self.ml_predictions = []
        self.llm_decisions = []
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        
        # Backtest loop
        logger.info("Running backtest loop...")
        signals_generated = 0
        trades_attempted = 0
        trades_executed = 0
        
        for idx, row in test_df_with_predictions.iterrows():
            current_price = row['close']
            
            # Update existing positions
            self._update_positions(row)
            
            # Check for ML signal
            if row.get('ml_signal', 0) == 1:
                signals_generated += 1
                
                # Prepare ML prediction info
                ml_prediction = {
                    'prediction': int(row['ml_prediction']),
                    'probability': float(row['ml_probability']),
                    'signal': int(row['ml_signal'])
                }
                
                self.ml_predictions.append({
                    'timestamp': idx,
                    'symbol': symbol,
                    'price': current_price,
                    **ml_prediction
                })
                
                # Check position limits
                if not self.check_position_limits():
                    continue
                
                # LLM decision gate
                if use_llm_gate:
                    trades_attempted += 1
                    
                    # Prepare indicators for LLM
                    indicators = {
                        'ema_fast': row.get('ema_fast', 0),
                        'ema_slow': row.get('ema_slow', 0),
                        'rsi': row.get('rsi', 50),
                        'atr': row.get('atr', 0),
                        'macd': row.get('macd', 0),
                        'macd_signal': row.get('macd_signal', 0),
                    }
                    
                    # Evaluate with LLM
                    decision = self.llm_gate.evaluate_candidate(
                        symbol=symbol,
                        current_price=current_price,
                        indicators=indicators,
                        ml_prediction=ml_prediction
                    )
                    
                    self.llm_decisions.append({
                        'timestamp': idx,
                        'symbol': symbol,
                        'price': current_price,
                        **decision
                    })
                    
                    if not decision['approve_trade']:
                        continue  # LLM rejected
                    
                    llm_confidence = decision['confidence']
                else:
                    # Skip LLM gate
                    trades_attempted += 1
                    llm_confidence = ml_prediction['probability']
                
                # Execute trade
                atr = row.get('atr', current_price * 0.02)
                position_size = self.calculate_position_size(current_price, llm_confidence, atr)
                stop_loss, take_profit = self.calculate_stops(current_price, 'long', atr)
                
                position = {
                    'entry_time': idx,
                    'entry_price': current_price,
                    'size': position_size,
                    'direction': 'long',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'ml_confidence': ml_prediction['probability'],
                    'llm_confidence': llm_confidence if use_llm_gate else None,
                    'status': 'open',
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': 0
                }
                
                self.positions.append(position)
                trades_executed += 1
            
            # Track equity
            self.equity_curve.append(self.capital)
        
        # Close any remaining positions
        self._close_all_positions(test_df_with_predictions.iloc[-1])
        
        logger.info(f"Backtest complete: {signals_generated} signals, {trades_attempted} attempted, {trades_executed} executed")
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(test_df_with_predictions)
        
        return {
            'config': self.config,
            'symbol': symbol,
            'model_name': model_name,
            'use_llm_gate': use_llm_gate,
            'train_size': train_size,
            'train_candles': len(train_df),
            'test_candles': len(test_df),
            'signals_generated': signals_generated,
            'trades_attempted': trades_attempted,
            'trades_executed': trades_executed,
            'metrics': self.metrics,
            'equity_curve': self.equity_curve,
            'positions': self.positions,
            'ml_predictions': self.ml_predictions,
            'llm_decisions': self.llm_decisions,
        }
    
    def _update_positions(self, row: pd.Series):
        """Update open positions and check for exits"""
        current_price = row['close']
        high = row['high']
        low = row['low']
        
        for position in self.positions:
            if position['status'] != 'open':
                continue
            
            # Check stops
            if position['direction'] == 'long':
                if low <= position['stop_loss']:
                    # Stop loss hit
                    self._close_position(position, position['stop_loss'], row.name, 'stop_loss')
                elif high >= position['take_profit']:
                    # Take profit hit
                    self._close_position(position, position['take_profit'], row.name, 'take_profit')
            else:  # short
                if high >= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], row.name, 'stop_loss')
                elif low <= position['take_profit']:
                    self._close_position(position, position['take_profit'], row.name, 'take_profit')
    
    def _close_position(self, position: Dict, exit_price: float, exit_time, reason: str):
        """Close a position and calculate P&L"""
        position['exit_price'] = exit_price
        position['exit_time'] = exit_time
        position['exit_reason'] = reason
        position['status'] = 'closed'
        
        # Calculate P&L
        if position['direction'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:  # short
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        position['pnl'] = pnl
        position['return'] = pnl / (position['entry_price'] * position['size'])
        
        # Update capital
        self.capital += pnl
        
        # Update peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        self.trades.append(position.copy())
    
    def _close_all_positions(self, last_row: pd.Series):
        """Close all remaining open positions at market price"""
        for position in self.positions:
            if position['status'] == 'open':
                self._close_position(position, last_row['close'], last_row.name, 'end_of_backtest')
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive backtest metrics"""
        metrics = {}
        
        # Trade metrics
        if len(self.trades) > 0:
            pnls = [t['pnl'] for t in self.trades]
            returns = [t['return'] for t in self.trades]
            
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            metrics['total_trades'] = len(self.trades)
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            metrics['win_rate'] = len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0
            
            metrics['total_pnl'] = sum(pnls)
            metrics['avg_pnl'] = np.mean(pnls)
            metrics['avg_win'] = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            metrics['avg_loss'] = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Returns
            metrics['total_return'] = (self.capital - self.initial_capital) / self.initial_capital
            metrics['avg_return'] = np.mean(returns)
            
            # Sharpe ratio (simplified - using trade returns)
            if len(returns) > 1:
                metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                metrics['sharpe_ratio'] = 0
            
            # Drawdown
            equity_array = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            metrics['max_drawdown'] = np.min(drawdown)
            
            # Exit reasons
            exit_reasons = {}
            for trade in self.trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            metrics['exit_reasons'] = exit_reasons
        else:
            metrics['total_trades'] = 0
            metrics['win_rate'] = 0
            metrics['total_pnl'] = 0
            metrics['total_return'] = 0
            metrics['profit_factor'] = 0
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
        
        # ML metrics
        if len(self.ml_predictions) > 0:
            metrics['ml_signals'] = len(self.ml_predictions)
            metrics['ml_avg_confidence'] = np.mean([p['probability'] for p in self.ml_predictions])
        else:
            metrics['ml_signals'] = 0
            metrics['ml_avg_confidence'] = 0
        
        # LLM metrics
        if len(self.llm_decisions) > 0:
            approved = [d for d in self.llm_decisions if d['approve_trade']]
            metrics['llm_decisions'] = len(self.llm_decisions)
            metrics['llm_approved'] = len(approved)
            metrics['llm_approval_rate'] = len(approved) / len(self.llm_decisions)
            metrics['llm_avg_confidence'] = np.mean([d['confidence'] for d in approved]) if approved else 0
        else:
            metrics['llm_decisions'] = 0
            metrics['llm_approved'] = 0
            metrics['llm_approval_rate'] = 0
            metrics['llm_avg_confidence'] = 0
        
        # Model metrics (from training)
        if hasattr(self.model_engine, 'training_metrics'):
            metrics['model_metrics'] = self.model_engine.training_metrics
        
        return metrics
    
    def print_results(self, results: Dict):
        """Print formatted backtest results"""
        print("\n" + "="*80)
        print(f"AI STRATEGY BACKTEST RESULTS - {results['symbol']}")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Model: {results['model_name']}")
        print(f"  LLM Gate: {'Enabled' if results['use_llm_gate'] else 'Disabled'}")
        print(f"  Train/Test Split: {results['train_size']:.0%} / {1-results['train_size']:.0%}")
        print(f"  Train Candles: {results['train_candles']}")
        print(f"  Test Candles: {results['test_candles']}")
        
        metrics = results['metrics']
        
        print(f"\nML Performance:")
        print(f"  Signals Generated: {results['signals_generated']}")
        print(f"  Average ML Confidence: {metrics['ml_avg_confidence']:.2%}")
        
        if results['use_llm_gate']:
            print(f"\nLLM Gate:")
            print(f"  Trades Evaluated: {metrics['llm_decisions']}")
            print(f"  Trades Approved: {metrics['llm_approved']}")
            print(f"  Approval Rate: {metrics['llm_approval_rate']:.2%}")
            print(f"  Average LLM Confidence: {metrics['llm_avg_confidence']:.2%}")
        
        print(f"\nTrading Performance:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Winning Trades: {metrics['winning_trades']}")
        print(f"  Losing Trades: {metrics['losing_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        
        print(f"\nProfit & Loss:")
        print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Average P&L per Trade: ${metrics.get('avg_pnl', 0):,.2f}")
        print(f"  Average Win: ${metrics.get('avg_win', 0):,.2f}")
        print(f"  Average Loss: ${metrics.get('avg_loss', 0):,.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        if 'exit_reasons' in metrics:
            print(f"\nExit Reasons:")
            for reason, count in metrics['exit_reasons'].items():
                print(f"  {reason}: {count}")
        
        print(f"\nFinal Capital: ${self.capital:,.2f}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Net Profit: ${self.capital - self.initial_capital:,.2f}")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict, filename: str):
        """Save backtest results to JSON file"""
        output_path = Path(self.config.backtest.data_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert non-serializable objects
        results_copy = results.copy()
        results_copy['config'] = {
            'model': results['model_name'],
            'use_llm_gate': results['use_llm_gate'],
            'initial_capital': self.config.risk.initial_capital,
        }
        
        # Convert timestamps to strings
        for position in results_copy['positions']:
            if hasattr(position['entry_time'], 'isoformat'):
                position['entry_time'] = position['entry_time'].isoformat()
            if position['exit_time'] and hasattr(position['exit_time'], 'isoformat'):
                position['exit_time'] = position['exit_time'].isoformat()
        
        for pred in results_copy['ml_predictions']:
            if hasattr(pred['timestamp'], 'isoformat'):
                pred['timestamp'] = pred['timestamp'].isoformat()
        
        for dec in results_copy['llm_decisions']:
            if hasattr(dec['timestamp'], 'isoformat'):
                dec['timestamp'] = dec['timestamp'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("AI Backtester Example")
    print("=" * 80)
    print("\nThis module requires historical data to run.")
    print("Use it programmatically:")
    print()
    print("  from ai_strategy.ai_backtester import AIBacktester")
    print("  from ai_strategy.config import AIStrategyConfig")
    print("  ")
    print("  config = AIStrategyConfig()")
    print("  backtester = AIBacktester(config)")
    print("  results = backtester.run_backtest(df, 'BTCUSDT')")
    print("  backtester.print_results(results)")
    print()
