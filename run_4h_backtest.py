#!/usr/bin/env python3
"""
4-Hour Timeframe Backtest
=========================

Testing on 4h timeframe for:
- Clearer trends (less noise)
- Better follow-through
- Higher win rate
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


def run_4h_backtest():
    """Test strategy on 4h timeframe"""
    
    print("="*80)
    print("📊 4-HOUR TIMEFRAME BACKTEST")
    print("="*80)
    
    print("\n🎯 Strategy: Test clearer trends with less noise")
    print("  ✓ Timeframe: 4h (was 1h)")
    print("  ✓ Days: 180 (more history)")
    print("  ✓ Prediction threshold: 0.55")
    print("  ✓ Stop loss: 2.5x ATR (wider)")
    print("  ✓ Position size: 2% (conservative)")
    
    symbol = "BTCUSDT"
    days = 180
    timeframe = "4h"
    
    # Fetch data
    print(f"\n📊 Fetching {days} days of {timeframe} data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_historical_data(symbol, timeframe, start_date, end_date)
    print(f"✅ Loaded {len(df)} candles ({timeframe})")
    
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
    
    # Configure
    config = AIStrategyConfig()
    
    # Model
    config.model.prediction_threshold = 0.55
    config.model.label_threshold = 0.007  # 0.7% move (4h needs larger moves)
    
    # Risk - more conservative on 4h
    config.risk.max_position_size_percent = 0.02  # 2% (was 3%)
    config.risk.stop_loss_atr_multiplier = 2.5  # Wider stops
    config.risk.take_profit_atr_multiplier = 5.0  # 2:1 R/R
    
    # LLM: Still disabled
    config.llm.provider = "mock"
    config.llm.use_llm_gate = False
    
    # Run backtest
    print(f"\n🤖 Running {timeframe} backtest...")
    backtester = AIBacktester(config)
    
    results = backtester.run_backtest(
        df=df,
        symbol=symbol,
        model_name="random_forest",
        use_llm_gate=False,
        train_size=0.7
    )
    
    # Print results
    backtester.print_results(results)
    
    # Comparison analysis
    print("\n" + "="*80)
    print("📊 4H vs 1H COMPARISON")
    print("="*80)
    
    metrics = results['metrics']
    
    print(f"\n📈 Trading Performance:")
    print(f"  Trades Executed: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    
    # Compare with 1h results
    print(f"\n🔄 vs 1h Timeframe:")
    print(f"  1h Win Rate: 22.22%")
    print(f"  4h Win Rate: {metrics['win_rate']:.2%}")
    
    if metrics['win_rate'] > 0.30:
        improvement = metrics['win_rate'] - 0.2222
        print(f"  ✅ Improvement: +{improvement:.2%}")
        if metrics['win_rate'] > 0.40:
            print(f"  🎉 EXCELLENT! Win rate > 40%")
        elif metrics['win_rate'] > 0.35:
            print(f"  👍 GOOD! Win rate > 35%")
        else:
            print(f"  📈 BETTER! But needs more tuning")
    else:
        print(f"  ⚠️  Still below 30% - try ADX filter")
    
    # Model quality
    if 'model_metrics' in metrics and 'random_forest' in metrics['model_metrics']:
        rf_metrics = metrics['model_metrics']['random_forest']
        print(f"\n🤖 ML Model (4h):")
        print(f"  Accuracy: {rf_metrics['accuracy']:.2%}")
        print(f"  AUC: {rf_metrics['auc']:.4f}")
        
        # Compare
        print(f"\n🔄 vs 1h ML Performance:")
        print(f"  1h AUC: 0.6567")
        print(f"  4h AUC: {rf_metrics['auc']:.4f}")
        
        if rf_metrics['auc'] > 0.6567:
            print(f"  ✅ 4h has better predictions!")
        elif rf_metrics['auc'] > 0.60:
            print(f"  👍 Both timeframes have good AUC")
        else:
            print(f"  ⚠️  4h AUC dropped - might need more data")
    
    # Risk metrics
    print(f"\n💰 Risk/Reward:")
    if metrics['total_trades'] > 0:
        print(f"  Average Win: ${metrics['avg_win']:.2f}")
        print(f"  Average Loss: ${abs(metrics['avg_loss']):.2f}")
        if metrics['avg_win'] > 0 and metrics['avg_loss'] != 0:
            rr_ratio = metrics['avg_win'] / abs(metrics['avg_loss'])
            print(f"  Win/Loss Ratio: {rr_ratio:.2f}:1")
            
            # Calculate breakeven win rate
            if rr_ratio > 0:
                breakeven = 1 / (rr_ratio + 1)
                print(f"  Breakeven Win Rate: {breakeven:.2%}")
                
                if metrics['win_rate'] > breakeven:
                    print(f"  ✅ Above breakeven! Strategy is profitable")
                else:
                    print(f"  ⚠️  Below breakeven - need higher win rate")
    
    return results, backtester


if __name__ == "__main__":
    try:
        results, backtester = run_4h_backtest()
        print("\n✅ 4h backtest complete!")
        
        metrics = results['metrics']
        
        print("\n🎯 Next Steps:")
        if metrics['win_rate'] > 0.40:
            print("1. ✅ 4h timeframe works well!")
            print("2. → Enable LLM gate with relaxed rules")
            print("3. → Test on other symbols (ETHUSDT, SOLUSDT)")
            print("4. → Run walk-forward optimization")
        elif metrics['win_rate'] > 0.30:
            print("1. 👍 4h is better than 1h")
            print("2. → Add ADX filter (only trade when trending)")
            print("3. → Add entry confirmation (wait for breakout)")
            print("4. → Widen stops to 3.0x ATR")
        else:
            print("1. ⚠️  Still struggling")
            print("2. → Add ADX filter immediately")
            print("3. → Try 1d timeframe (even clearer trends)")
            print("4. → Review feature engineering (add volume)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
