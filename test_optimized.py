"""
Test optimized conservative strategies
"""

import sys
from backtest_deterministic import DeterministicBacktester, save_results
from optimized_strategies import get_optimized_strategies
from data_fetcher import BinanceDataFetcher
from datetime import datetime


def run_optimized_backtests(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 180
):
    """Run backtests for optimized strategies"""
    
    print("=" * 80)
    print("🎯 OPTIMIZED CONSERVATIVE STRATEGIES BACKTEST")
    print("=" * 80)
    print(f"\nSymbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Period: {days} days")
    print(f"Capital: $10,000")
    print(f"Risk per trade: 1% (max $100)")
    print(f"Stop Loss: 2% | Take Profit: 4% (2:1 R/R)")
    print(f"Commission: 0.1% per side")
    print(f"Slippage: 0.05%")
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
    backtester = DeterministicBacktester(
        initial_capital=10000,
        risk_per_trade=0.01,  # 1%
        commission=0.001,      # 0.1%
        slippage=0.0005        # 0.05%
    )
    
    strategies = get_optimized_strategies()
    results = []
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*80}")
        print(f"Testing Strategy {i}/{len(strategies)}: {strategy.name}")
        print(f"{'='*80}")
        
        result = backtester.backtest_strategy(
            strategy, 
            df.copy(),
            stop_loss_pct=0.02,    # 2% stop
            take_profit_pct=0.04   # 4% target
        )
        results.append(result)
        
        # Print results
        m = result['metrics']
        print(f"\n📊 Results:")
        print(f"   Total Trades: {m['total_trades']}")
        print(f"   Win Rate: {m['win_rate']}%")
        print(f"   Profit Factor: {m['profit_factor']}")
        print(f"   Total Return: {m['total_return_pct']}%")
        print(f"   Max Drawdown: {m['max_drawdown_pct']}%")
        print(f"   Avg Win: ${m['avg_win']:.2f}")
        print(f"   Avg Loss: ${m['avg_loss']:.2f}")
        print(f"   Win/Loss Ratio: {m['win_loss_ratio']}")
        print(f"   Sharpe Ratio: {m['sharpe_ratio']}")
        print(f"   Final Capital: ${result['final_capital']:.2f}")
        
        # Assessment
        if m['total_return_pct'] > 5 and m['win_rate'] > 35:
            print("\n   ✅ ✅ EXCELLENT STRATEGY!")
        elif m['total_return_pct'] > 2 and m['win_rate'] > 30:
            print("\n   ✅ PROFITABLE STRATEGY!")
        elif m['total_return_pct'] > 0:
            print("\n   ⚠️  Slightly profitable")
        elif m['total_return_pct'] > -1:
            print("\n   ⚠️  Near break-even")
        else:
            print("\n   ❌ Needs improvement")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 STRATEGY COMPARISON")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x['metrics']['total_return_pct'], reverse=True)
    
    print(f"\n{'#':<4}{'Strategy':<30}{'Return':<12}{'WR':<10}{'Trades':<10}{'PF':<8}{'Sharpe'}")
    print("-" * 90)
    
    for i, result in enumerate(sorted_results, 1):
        m = result['metrics']
        marker = "🏆" if i == 1 else "✅" if m['total_return_pct'] > 0 else "❌"
        print(f"{marker} {i:<3}{result['strategy']:<29}{m['total_return_pct']:>9}%  "
              f"{m['win_rate']:>7}%  {m['total_trades']:>8}  {m['profit_factor']:>6}  {m['sharpe_ratio']:>6}")
    
    # Best strategy details
    if sorted_results and sorted_results[0]['metrics']['total_return_pct'] > 0:
        print("\n" + "=" * 80)
        print("🏆 BEST STRATEGY DETAILS")
        print("=" * 80)
        best = sorted_results[0]
        m = best['metrics']
        
        print(f"\nStrategy: {best['strategy']}")
        print(f"Final Capital: ${best['final_capital']:.2f}")
        print(f"Total Return: {m['total_return_pct']:.2f}%")
        print(f"Win Rate: {m['win_rate']}%")
        print(f"Profit Factor: {m['profit_factor']}")
        print(f"Max Drawdown: {m['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {m['sharpe_ratio']}")
        print(f"\nAverage Win: ${m['avg_win']:.2f}")
        print(f"Average Loss: ${m['avg_loss']:.2f}")
        print(f"Win/Loss Ratio: {m['win_loss_ratio']:.2f}:1")
        
        # Show some trades
        if best['trades']:
            print(f"\n📝 Sample Trades (last 5):")
            for trade in best['trades'][-5:]:
                emoji = "✅" if trade['pnl'] > 0 else "❌"
                print(f"   {emoji} {trade['entry_date'].strftime('%Y-%m-%d')}: "
                      f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} = "
                      f"${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%) - {trade['exit_reason']}")
    
    return results


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    
    results = run_optimized_backtests(symbol, interval, days)
