#!/usr/bin/env python3
"""
Simple AI Backtest Runner
==========================

Simplified runner that handles import paths correctly.
"""

import sys
from pathlib import Path

# Set up paths properly
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Now we can import
from backtesting.data_fetcher import get_historical_data
from backtesting import indicators
from ai_strategy.config import AIStrategyConfig
from ai_strategy.ai_backtester import AIBacktester
from datetime import datetime, timedelta
import argparse


def run_backtest(symbol: str = "BTCUSDT", days: int = 90, model: str = "random_forest"):
    """Run AI strategy backtest"""
    
    print("="*80)
    print(f"🚀 AI STRATEGY BACKTEST - {symbol}")
    print("="*80)
    
    # Fetch data
    print(f"\n📊 Fetching {days} days of data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_historical_data(symbol, "1h", start_date, end_date)
    print(f"✅ Loaded {len(df)} candles")
    
    # Calculate indicators
    print("\n📈 Calculating indicators...")
    df['ema_fast'] = indicators.compute_ema(df['close'], 20)
    df['ema_slow'] = indicators.compute_ema(df['close'], 50)
    df['ema_200'] = indicators.compute_ema(df['close'], 200)
    
    df['atr'] = indicators.compute_atr(df, 14)
    df['rsi'] = indicators.compute_rsi(df['close'], 14)
    
    df['macd'], df['macd_signal'], df['macd_histogram'] = indicators.compute_macd(df['close'], 12, 26, 9)
    
    bb_upper, bb_middle, bb_lower = indicators.compute_bollinger_bands(df['close'], 20, 2.0)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    df = df.dropna()
    print(f"✅ {len(df)} candles with indicators")
    
    # Configure
    config = AIStrategyConfig()
    config.llm.provider = "mock"
    config.llm.use_llm_gate = True
    
    # Run backtest
    print(f"\n🤖 Running AI backtest with {model}...")
    backtester = AIBacktester(config)
    
    results = backtester.run_backtest(
        df=df,
        symbol=symbol,
        model_name=model,
        use_llm_gate=True,
        train_size=0.7
    )
    
    # Print results
    backtester.print_results(results)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{symbol}_{model}_{timestamp}.json"
    backtester.save_results(results, filename)
    
    return results, backtester


def compare_models(symbol: str = "BTCUSDT", days: int = 90):
    """Compare different ML models"""
    
    print("="*80)
    print(f"🔬 MODEL COMPARISON - {symbol}")
    print("="*80)
    
    # Fetch data once
    print(f"\n📊 Fetching {days} days of data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_historical_data(symbol, "1h", start_date, end_date)
    
    # Indicators
    print(f"📈 Calculating indicators...")
    df['ema_fast'] = indicators.compute_ema(df['close'], 20)
    df['ema_slow'] = indicators.compute_ema(df['close'], 50)
    df['ema_200'] = indicators.compute_ema(df['close'], 200)
    df['atr'] = indicators.compute_atr(df, 14)
    df['rsi'] = indicators.compute_rsi(df['close'], 14)
    
    df['macd'], df['macd_signal'], df['macd_histogram'] = indicators.compute_macd(df['close'], 12, 26, 9)
    
    bb_upper, bb_middle, bb_lower = indicators.compute_bollinger_bands(df['close'], 20, 2.0)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    df = df.dropna()
    
    models = ["random_forest", "xgboost", "logistic_regression"]
    results_list = []
    
    config = AIStrategyConfig()
    config.llm.provider = "mock"
    
    for model_name in models:
        print(f"\n{'─'*80}")
        print(f"Testing {model_name.upper()}")
        print(f"{'─'*80}")
        
        try:
            backtester = AIBacktester(config)
            results = backtester.run_backtest(
                df=df,
                symbol=symbol,
                model_name=model_name,
                use_llm_gate=True,
                train_size=0.7
            )
            
            results_list.append((model_name, results))
            
            metrics = results['metrics']
            print(f"\n📊 {model_name.upper()} Summary:")
            print(f"  Total Return: {metrics['total_return']:.2%}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"  Trades: {metrics['total_trades']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Return':>12} {'Win Rate':>12} {'Sharpe':>10} {'Trades':>10}")
    print("─"*80)
    
    for model_name, results in results_list:
        metrics = results['metrics']
        print(f"{model_name:<20} {metrics['total_return']:>11.2%} "
              f"{metrics['win_rate']:>11.2%} {metrics['sharpe_ratio']:>9.2f} "
              f"{metrics['total_trades']:>10}")
    
    if results_list:
        best = max(results_list, key=lambda x: x[1]['metrics']['total_return'])
        print(f"\n🏆 Best: {best[0].upper()} (Return: {best[1]['metrics']['total_return']:.2%})")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(description="Simple AI Backtest Runner")
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=90, help='Days of data')
    parser.add_argument('--model', default='random_forest', 
                        choices=['random_forest', 'xgboost', 'logistic_regression'])
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_models(args.symbol, args.days)
        else:
            run_backtest(args.symbol, args.days, args.model)
        
        print("\n✅ Complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
