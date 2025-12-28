"""
Test Refined MACD Strategies
Focus on pushing MACD from -0.23% to profitability
"""

import sys
from backtest_deterministic import DeterministicBacktester
from refined_macd import get_refined_macd_strategies
from data_fetcher import BinanceDataFetcher


def test_refined_macd(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 180
):
    """Test refined MACD strategies"""
    
    print("=" * 80)
    print("🎯 REFINED MACD STRATEGY TESTING")
    print("=" * 80)
    print(f"\n📊 Original MACD Performance:")
    print(f"   Return: -0.23%")
    print(f"   Win Rate: 35.9%")
    print(f"   Profit Factor: 1.09")
    print(f"   Trades: 39")
    print(f"   Status: CLOSEST TO BREAK-EVEN")
    print(f"\n🎯 Goal: Push into profitability with refinements")
    print("\n" + "=" * 80)
    print(f"\nSymbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Period: {days} days")
    print(f"Capital: $10,000")
    print(f"Risk: 1% per trade")
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
    
    # Run tests
    backtester = DeterministicBacktester()
    strategies = get_refined_macd_strategies()
    
    results = []
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'='*80}")
        print(f"Testing {i}/{len(strategies)}: {strategy.name}")
        print(f"{'='*80}")
        
        result = backtester.backtest_strategy(strategy, df.copy())
        results.append(result)
        
        m = result['metrics']
        print(f"\n📊 Results:")
        print(f"   Total Trades: {m['total_trades']}")
        print(f"   Win Rate: {m['win_rate']}%")
        print(f"   Profit Factor: {m['profit_factor']}")
        print(f"   Total Return: {m['total_return_pct']}%")
        print(f"   Max Drawdown: {m['max_drawdown_pct']}%")
        print(f"   Sharpe Ratio: {m['sharpe_ratio']}")
        print(f"   Final Capital: ${result['final_capital']:.2f}")
        
        # Compare to original
        improvement = m['total_return_pct'] - (-0.23)
        print(f"\n   📈 vs Original MACD: {improvement:+.2f}%")
        
        if m['total_return_pct'] > 0:
            print(f"   ✅✅ PROFITABLE! (+{m['total_return_pct']:.2f}%)")
        elif m['total_return_pct'] > -0.23:
            print(f"   ✅ IMPROVED! (Better than original)")
        elif m['total_return_pct'] == -0.23:
            print(f"   ⚠️  Same as original")
        else:
            print(f"   ❌ Worse than original")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY - MACD REFINEMENTS")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x['metrics']['total_return_pct'], reverse=True)
    
    print(f"\n{'#':<4}{'Strategy':<30}{'Return':<12}{'vs Orig':<12}{'WR':<10}{'Trades':<10}{'PF'}")
    print("-" * 90)
    
    original_return = -0.23
    
    for i, result in enumerate(sorted_results, 1):
        m = result['metrics']
        improvement = m['total_return_pct'] - original_return
        marker = "🏆" if m['total_return_pct'] > 0 else "✅" if improvement > 0 else "❌"
        
        print(f"{marker} {i:<3}{result['strategy']:<29}{m['total_return_pct']:>9}%  "
              f"{improvement:>9.2f}%  {m['win_rate']:>7}%  {m['total_trades']:>8}  {m['profit_factor']:>6}")
    
    # Best strategy
    if sorted_results:
        best = sorted_results[0]
        m = best['metrics']
        
        print("\n" + "=" * 80)
        if m['total_return_pct'] > 0:
            print("🏆 SUCCESS! FOUND PROFITABLE MACD VARIATION")
        else:
            print("📊 BEST MACD REFINEMENT")
        print("=" * 80)
        
        print(f"\nStrategy: {best['strategy']}")
        print(f"Return: {m['total_return_pct']:.2f}% (vs -0.23% original)")
        print(f"Improvement: {m['total_return_pct'] - original_return:+.2f}%")
        print(f"Win Rate: {m['win_rate']}% (vs 35.9% original)")
        print(f"Profit Factor: {m['profit_factor']} (vs 1.09 original)")
        print(f"Trades: {m['total_trades']} (vs 39 original)")
        print(f"Max Drawdown: {m['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {m['sharpe_ratio']}")
        
        if m['total_return_pct'] > 0:
            print(f"\n✅ MACD IS NOW PROFITABLE!")
            print(f"   Original: -0.23% (lost money)")
            print(f"   Refined:  +{m['total_return_pct']:.2f}% (makes money)")
            print(f"   Swing:    +{m['total_return_pct'] + 0.23:.2f}% improvement")
        
        # Show trades
        if best['trades']:
            print(f"\n📝 Last 5 Trades:")
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
    
    results = test_refined_macd(symbol, interval, days)
