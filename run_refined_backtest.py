#!/usr/bin/env python3
"""
Refined AI Strategy Backtest
=============================

Running improved strategy with:
- More data (180 days)
- Lower prediction threshold (0.55)
- Relaxed LLM gate
- Wider stops
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from backtesting.data_fetcher import get_historical_data
from backtesting import indicators
from ai_strategy.config import AIStrategyConfig
from ai_strategy.ai_backtester import AIBacktester
from datetime import datetime, timedelta


def run_refined_backtest():
    """Run refined strategy with improvements"""
    
    print("="*80)
    print("🔧 REFINED AI STRATEGY BACKTEST")
    print("="*80)
    
    # Configuration improvements
    print("\n⚙️  Configuration Changes:")
    print("  ✓ Training data: 180 days (was 90)")
    print("  ✓ Prediction threshold: 0.55 (was 0.60)")
    print("  ✓ LLM gate: DISABLED for testing")
    print("  ✓ Stop loss: 2.0x ATR (was 1.5x)")
    print("  ✓ Position size: 3% (was 2%)")
    
    symbol = "BTCUSDT"
    days = 180
    
    # Fetch more data
    print(f"\n📊 Fetching {days} days of data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_historical_data(symbol, "1h", start_date, end_date)
    print(f"✅ Loaded {len(df)} candles")
    
    # Indicators
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
    
    # Configure with improvements
    config = AIStrategyConfig()
    
    # Model improvements
    config.model.prediction_threshold = 0.55  # Lower threshold (more signals)
    config.model.label_threshold = 0.005  # 0.5% move (was 0.1%)
    
    # Risk improvements  
    config.risk.max_position_size_percent = 0.03  # 3% (was 2%)
    config.risk.stop_loss_atr_multiplier = 2.0  # Wider stops
    config.risk.take_profit_atr_multiplier = 4.0  # 2:1 R/R
    
    # LLM: Disable for now to isolate ML performance
    config.llm.provider = "mock"
    config.llm.use_llm_gate = False
    
    # Run backtest
    print(f"\n🤖 Running refined backtest...")
    backtester = AIBacktester(config)
    
    results = backtester.run_backtest(
        df=df,
        symbol=symbol,
        model_name="random_forest",
        use_llm_gate=False,  # Disabled
        train_size=0.7
    )
    
    # Print results
    backtester.print_results(results)
    
    # Analysis
    print("\n" + "="*80)
    print("📊 REFINED STRATEGY ANALYSIS")
    print("="*80)
    
    metrics = results['metrics']
    
    print(f"\n✅ Improvements:")
    print(f"  Signals Generated: {results['signals_generated']}")
    print(f"  Trades Executed: {metrics['total_trades']}")
    print(f"  ML Average Confidence: {metrics['ml_avg_confidence']:.2%}")
    
    if metrics['total_trades'] >= 10:
        print(f"\n📈 Trading Performance:")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        
        if metrics['win_rate'] > 0.45:
            print(f"\n🎉 WIN RATE > 45% - Strategy shows promise!")
        elif metrics['win_rate'] > 0.40:
            print(f"\n👍 WIN RATE > 40% - Decent performance, needs tuning")
        else:
            print(f"\n⚠️  WIN RATE < 40% - Needs more refinement")
    else:
        print(f"\n⚠️  Only {metrics['total_trades']} trades - need more data or lower threshold")
    
    # Model quality
    if 'model_metrics' in metrics and 'random_forest' in metrics['model_metrics']:
        rf_metrics = metrics['model_metrics']['random_forest']
        print(f"\n🤖 ML Model Quality:")
        print(f"  Accuracy: {rf_metrics['accuracy']:.2%}")
        print(f"  AUC: {rf_metrics['auc']:.4f}")
        
        if rf_metrics['auc'] > 0.60:
            print(f"  ✅ AUC > 0.60 - Good predictive power")
        elif rf_metrics['auc'] > 0.55:
            print(f"  👍 AUC > 0.55 - Acceptable predictive power")
        else:
            print(f"  ⚠️  AUC < 0.55 - Weak predictive power, add more features")
    
    return results, backtester


if __name__ == "__main__":
    try:
        results, backtester = run_refined_backtest()
        print("\n✅ Backtest complete!")
        print("\nNext steps:")
        print("1. If performance good → enable LLM gate with lower threshold")
        print("2. If still weak → add volume features, try 4h timeframe")
        print("3. Run parameter optimization")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
