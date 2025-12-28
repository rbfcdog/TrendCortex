#!/usr/bin/env python3
"""
Quick demo script showing backtesting system capabilities.
Runs multiple backtests with different parameters to demonstrate the system.
"""

import sys
from datetime import datetime, timedelta
from run_backtest import run_single_backtest
from config import APPROVED_SYMBOLS

print("=" * 80)
print("TRENDCORTEX BACKTESTING SYSTEM - QUICK DEMO")
print("=" * 80)
print()

# Demo 1: Recent week with faster EMAs for more trades
print("📊 Demo 1: Recent 14 days - BTCUSDT on 1h with faster EMAs (10/20)")
print("-" * 80)

result = run_single_backtest(
    symbol="BTCUSDT",
    interval="1h",
    start_date=datetime.now() - timedelta(days=14),
    end_date=datetime.now(),
    strategy_params={
        'fast_ema': 10,
        'slow_ema': 20,
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'min_atr': 0.001
    },
    use_cache=True
)

if result and result.get('trades', 0) > 0:
    print(f"\n✅ Executed {result['trades']} trades")
    print(f"   Win Rate: {result['win_rate']:.1f}%")
    print(f"   Total P&L: ${result['total_pnl']:.2f} ({result['total_return']:.2f}%)")
else:
    print("\n⚠️  No trades executed in this period")

print("\n" + "=" * 80)
print("📊 Demo 2: 30 days - Multiple symbols on 2h timeframe")
print("-" * 80)

test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
results_summary = []

for symbol in test_symbols:
    result = run_single_backtest(
        symbol=symbol,
        interval="2h",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        strategy_params={
            'fast_ema': 12,
            'slow_ema': 26,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_atr': 0.001
        },
        use_cache=True
    )
    
    if result:
        results_summary.append({
            'symbol': symbol,
            'trades': result.get('trades', 0),
            'win_rate': result.get('win_rate', 0),
            'pnl': result.get('total_pnl', 0),
            'return': result.get('total_return', 0)
        })

print("\n📈 Results Summary:")
print(f"{'Symbol':<10} {'Trades':>8} {'Win Rate':>10} {'P&L':>12} {'Return':>10}")
print("-" * 60)
for r in results_summary:
    print(f"{r['symbol']:<10} {r['trades']:>8} {r['win_rate']:>9.1f}% ${r['pnl']:>10.2f} {r['return']:>9.2f}%")

total_trades = sum(r['trades'] for r in results_summary)
total_pnl = sum(r['pnl'] for r in results_summary)
avg_win_rate = sum(r['win_rate'] for r in results_summary if r['trades'] > 0) / len([r for r in results_summary if r['trades'] > 0]) if any(r['trades'] > 0 for r in results_summary) else 0

print("-" * 60)
print(f"{'TOTAL':<10} {total_trades:>8} {avg_win_rate:>9.1f}% ${total_pnl:>10.2f}")

print("\n" + "=" * 80)
print("✅ DEMO COMPLETE!")
print("=" * 80)
print()
print("💡 To run your own backtests:")
print()
print("  # Single symbol, 7 days")
print("  python run_backtest.py --symbols BTCUSDT --interval 1h --days 7")
print()
print("  # Multiple symbols, 30 days, save results")
print("  python run_backtest.py --symbols BTCUSDT ETHUSDT --interval 4h --days 30 --save-results")
print()
print("  # All symbols with custom EMA periods")
print("  python run_backtest.py --all-symbols --interval 1d --days 90 --fast-ema 12 --slow-ema 26")
print()
print("  # Custom date range")
print("  python run_backtest.py --symbols SOLUSDT --interval 2h --date-range 2025-11-01 2025-12-26")
print()
print("📖 For more information: cat README.md")
print("=" * 80)
