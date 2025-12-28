"""
Backtest Deterministic Strategies
Test pure technical analysis strategies without ML
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from pathlib import Path

from deterministic_strategies import get_all_strategies
from data_fetcher import BinanceDataFetcher


class DeterministicBacktester:
    """Backtest deterministic trading strategies"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005  # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        
    def calculate_position_size(
        self, 
        capital: float, 
        entry_price: float, 
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk
        Risk = capital * risk_per_trade
        Position Size = Risk / (Entry - Stop Loss)
        """
        risk_amount = capital * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0
            
        position_size = risk_amount / risk_per_unit
        
        # Don't use more than 10% of capital per trade
        max_position = capital * 0.10
        position_value = position_size * entry_price
        
        if position_value > max_position:
            position_size = max_position / entry_price
            
        return position_size
    
    def backtest_strategy(
        self, 
        strategy, 
        df: pd.DataFrame,
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04  # 4% take profit (2:1 R/R)
    ) -> Dict:
        """
        Backtest a single strategy
        
        Returns:
            Dictionary with results including trades, metrics, equity curve
        """
        df = strategy.generate_signals(df)
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        stop_loss = 0
        take_profit = 0
        
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(df)):
            current_price = df.iloc[i]['close']
            current_date = df.iloc[i]['timestamp']
            
            # Check if we're in a position
            if position > 0:
                # Check stop loss
                if df.iloc[i]['low'] <= stop_loss:
                    exit_price = stop_loss * (1 - self.slippage)  # Slippage on exit
                    pnl = (exit_price - entry_price) * position
                    pnl -= (entry_price * position * self.commission)  # Entry commission
                    pnl -= (exit_price * position * self.commission)  # Exit commission
                    
                    capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position)) * 100,
                        'exit_reason': 'stop_loss',
                        'capital': capital
                    })
                    
                    position = 0
                    equity_curve.append(capital)
                    continue
                
                # Check take profit
                if df.iloc[i]['high'] >= take_profit:
                    exit_price = take_profit * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * position
                    pnl -= (entry_price * position * self.commission)
                    pnl -= (exit_price * position * self.commission)
                    
                    capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position)) * 100,
                        'exit_reason': 'take_profit',
                        'capital': capital
                    })
                    
                    position = 0
                    equity_curve.append(capital)
                    continue
                
                # Check exit signal
                if df.iloc[i]['exit_signal'] == 1:
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * position
                    pnl -= (entry_price * position * self.commission)
                    pnl -= (exit_price * position * self.commission)
                    
                    capital += pnl
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position)) * 100,
                        'exit_reason': 'signal',
                        'capital': capital
                    })
                    
                    position = 0
                    equity_curve.append(capital)
            
            # Check entry signal
            if position == 0 and df.iloc[i]['signal'] == 1:
                entry_price = current_price * (1 + self.slippage)
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                
                position = self.calculate_position_size(capital, entry_price, stop_loss)
                entry_date = current_date
                
                # Deduct entry commission from capital immediately
                capital -= entry_price * position * self.commission
        
        # Close any open position at the end
        if position > 0:
            exit_price = df.iloc[-1]['close'] * (1 - self.slippage)
            pnl = (exit_price - entry_price) * position
            pnl -= (exit_price * position * self.commission)
            capital += pnl
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.iloc[-1]['timestamp'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * position)) * 100,
                'exit_reason': 'end_of_data',
                'capital': capital
            })
            equity_curve.append(capital)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, equity_curve)
        
        return {
            'strategy': strategy.name,
            'trades': trades,
            'metrics': metrics,
            'equity_curve': equity_curve,
            'final_capital': capital
        }
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return_pct': 0,
                'max_drawdown_pct': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'win_loss_ratio': 0,
                'sharpe_ratio': 0
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Return
        total_return_pct = ((equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown_pct = abs(drawdown.min())
        
        # Win/Loss stats
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe ratio (simplified)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_return_pct': round(total_return_pct, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }


def run_all_backtests(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 180
) -> List[Dict]:
    """
    Run backtests for all deterministic strategies
    
    Args:
        symbol: Trading pair
        interval: Timeframe
        days: Days of historical data
        
    Returns:
        List of results for each strategy
    """
    print("=" * 80)
    print("DETERMINISTIC STRATEGY BACKTESTING")
    print("=" * 80)
    print(f"\nSymbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Period: {days} days")
    print(f"Capital: $10,000")
    print(f"Risk per trade: 1%")
    print(f"Stop Loss: 2% | Take Profit: 4% (2:1 R/R)")
    print("\n" + "=" * 80)
    
    # Fetch data
    print("\n📊 Fetching market data...")
    fetcher = BinanceDataFetcher()
    df = fetcher.get_historical_klines(symbol, interval, days)
    
    if df is None or len(df) == 0:
        print("❌ Failed to fetch data")
        return []
    
    print(f"✅ Loaded {len(df)} candles")
    print(f"   From: {df['timestamp'].iloc[0]}")
    print(f"   To: {df['timestamp'].iloc[-1]}")
    
    # Run backtests
    backtester = DeterministicBacktester()
    strategies = get_all_strategies()
    
    results = []
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*80}")
        print(f"Testing Strategy {i}/{len(strategies)}: {strategy.name}")
        print(f"{'='*80}")
        
        result = backtester.backtest_strategy(strategy, df.copy())
        results.append(result)
        
        # Print results
        metrics = result['metrics']
        print(f"\n📊 Results:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']}%")
        print(f"   Profit Factor: {metrics['profit_factor']}")
        print(f"   Total Return: {metrics['total_return_pct']}%")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']}%")
        print(f"   Avg Win: ${metrics['avg_win']:.2f}")
        print(f"   Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"   Win/Loss Ratio: {metrics['win_loss_ratio']}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']}")
        print(f"   Final Capital: ${result['final_capital']:.2f}")
        
        # Quick assessment
        if metrics['total_return_pct'] > 0 and metrics['win_rate'] > 30:
            print("\n   ✅ PROFITABLE STRATEGY!")
        elif metrics['total_return_pct'] > 0:
            print("\n   ⚠️  Profitable but low win rate")
        elif metrics['win_rate'] > 40:
            print("\n   ⚠️  Good win rate but needs optimization")
        else:
            print("\n   ❌ Needs improvement")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    
    # Sort by total return
    sorted_results = sorted(results, key=lambda x: x['metrics']['total_return_pct'], reverse=True)
    
    print(f"\n{'Rank':<6}{'Strategy':<30}{'Return':<12}{'Win Rate':<12}{'Trades':<10}{'P.Factor'}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results, 1):
        m = result['metrics']
        print(f"{i:<6}{result['strategy']:<30}{m['total_return_pct']:>10}%  {m['win_rate']:>9}%  {m['total_trades']:>8}  {m['profit_factor']:>7}")
    
    # Save results
    save_results(results, symbol, interval, days)
    
    return results


def save_results(results: List[Dict], symbol: str, interval: str, days: int):
    """Save backtest results to JSON"""
    output_dir = Path("backtest_results_deterministic")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"backtest_{symbol}_{interval}_{days}d_{timestamp}.json"
    
    # Convert to serializable format
    serializable_results = []
    for result in results:
        serializable_results.append({
            'strategy': result['strategy'],
            'metrics': result['metrics'],
            'final_capital': result['final_capital'],
            'trade_count': len(result['trades'])
        })
    
    with open(filename, 'w') as f:
        json.dump({
            'symbol': symbol,
            'interval': interval,
            'days': days,
            'test_date': timestamp,
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")


if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    
    results = run_all_backtests(symbol, interval, days)
    
    print("\n" + "=" * 80)
    print("🎯 BEST STRATEGY:")
    best = max(results, key=lambda x: x['metrics']['total_return_pct'])
    print(f"   {best['strategy']}")
    print(f"   Return: {best['metrics']['total_return_pct']}%")
    print(f"   Win Rate: {best['metrics']['win_rate']}%")
    print(f"   Profit Factor: {best['metrics']['profit_factor']}")
    print("=" * 80)
