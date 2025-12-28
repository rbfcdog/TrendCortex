"""
Test MACD parameter optimization
"""

import sys
from backtest_deterministic import DeterministicBacktester
from optimized_macd import get_macd_variants
from data_fetcher import BinanceDataFetcher


def test_macd_optimization(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 180
):
    print("=" * 80)
    print("🎯 MACD PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"\nOriginal MACD: -0.23%, 35.9% WR, PF 1.09")
    print(f"Goal: Find parameters that make it profitable")
    print("\n" + "=" * 80)
    
    # Fetch data
    print("\n📊 Fetching data...")
    fetcher = BinanceDataFetcher()
    df = fetcher.get_historical_klines(symbol, interval, days)
    
    if df is None:
        return []
    
    print(f"✅ Loaded {len(df)} candles")
    
    # Test variants
    backtester = DeterministicBacktester()
    variants = get_macd_variants()
    
    results = []
    
    for i, variant in enumerate(variants, 1):
        print(f"\n[{i}/{len(variants)}] Testing: {variant.name}", end=" ... ")
        
        result = backtester.backtest_strategy(variant, df.copy())
        results.append(result)
        
        m = result['metrics']
        status = "✅ PROFIT!" if m['total_return_pct'] > 0 else f"{m['total_return_pct']:.2f}%"
        print(f"{status} (WR: {m['win_rate']}%, Trades: {m['total_trades']})")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x['metrics']['total_return_pct'], reverse=True)
    
    print(f"\n{'#':<4}{'Variant':<35}{'Return':<12}{'WR':<10}{'Trades':<10}{'PF'}")
    print("-" * 85)
    
    for i, result in enumerate(sorted_results, 1):
        m = result['metrics']
        marker = "🏆" if m['total_return_pct'] > 0 else "⚠️" if m['total_return_pct'] > -0.5 else "❌"
        print(f"{marker} {i:<3}{result['strategy']:<34}{m['total_return_pct']:>9}%  "
              f"{m['win_rate']:>7}%  {m['total_trades']:>8}  {m['profit_factor']:>6}")
    
    # Best
    if sorted_results:
        best = sorted_results[0]
        m = best['metrics']
        
        print("\n" + "=" * 80)
        print("🏆 BEST MACD VARIANT")
        print("=" * 80)
        print(f"\nVariant: {best['strategy']}")
        print(f"Return: {m['total_return_pct']:.2f}%")
        print(f"Win Rate: {m['win_rate']}%")
        print(f"Profit Factor: {m['profit_factor']}")
        print(f"Trades: {m['total_trades']}")
        print(f"Sharpe: {m['sharpe_ratio']}")
        
        if m['total_return_pct'] > 0:
            print(f"\n✅✅ FOUND PROFITABLE MACD!")
        else:
            print(f"\n⚠️  Still not profitable, but best variant")
    
    return results


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    
    test_macd_optimization(symbol, interval, days)
