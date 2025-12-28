#!/usr/bin/env python3
"""
AI Strategy Demo Runner
=======================

Demonstrates the complete AI trading strategy pipeline:
1. Fetch historical data
2. Train ML models
3. Generate predictions
4. Evaluate with LLM gate
5. Show results

Usage:
    python run_ai_demo.py --symbol BTCUSDT --days 90 --train
    python run_ai_demo.py --symbol ETHUSDT --predict --model-timestamp 20251226_123456
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backtesting"))

# Import AI strategy components
from ai_strategy.config import AIStrategyConfig
from ai_strategy.model_engine import ModelEngine
from ai_strategy.llm_gate import LLMGate

# Import data fetcher - lazy import to avoid config issues
def get_historical_data(symbol, interval, start_date, end_date):
    """Wrapper for data fetcher with proper path handling"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "data_fetcher",
        str(project_root / "backtesting" / "data_fetcher.py")
    )
    data_fetcher = importlib.util.module_from_spec(spec)
    sys.modules["data_fetcher"] = data_fetcher
    spec.loader.exec_module(data_fetcher)
    return data_fetcher.get_historical_data(symbol, interval, start_date, end_date)


def train_models(symbol: str, days: int = 90, interval: str = "1h"):
    """
    Train ML models on historical data
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        days: Number of days of historical data
        interval: Candle interval (1h, 4h, 1d)
    """
    print("=" * 80)
    print(f"🤖 AI Strategy Training: {symbol}")
    print("=" * 80)
    
    # Setup
    config = AIStrategyConfig()
    engine = ModelEngine(config)
    
    # Fetch data
    print(f"\n📊 Fetching {days} days of {interval} data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_historical_data(symbol, interval, start_date, end_date)
    print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Train models
    print(f"\n🧠 Training ML models...")
    print(f"   - Warmup periods: {config.backtest.warmup_periods}")
    print(f"   - Train/test split: 70/30 (chronological)")
    print(f"   - Feature engineering: 20+ features")
    
    models = engine.train_models(df)
    
    # Show metrics
    print("\n📈 Model Performance (Test Set):")
    print("-" * 80)
    
    for name, metrics in engine.training_metrics.items():
        print(f"\n{name.upper()}:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  F1 Score:        {metrics['f1']:.4f}")
        print(f"  AUC:             {metrics['auc']:.4f}")
        print(f"  Trade Accuracy:  {metrics['trade_accuracy']:.4f}")
    
    # Feature importance
    print("\n🎯 Top 10 Most Important Features:")
    print("-" * 80)
    
    importance = engine.get_feature_importance(model_name="random_forest", top_n=10)
    if importance is not None and len(importance) > 0:
        for i, row in enumerate(importance, 1):
            feature = row[0]
            score = row[1]
            print(f"{i:2d}. {feature:30s} {score:.4f}")
    
    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = engine.save_models(suffix=f"_{symbol}_{timestamp}")
    print(f"\n💾 Models saved to: {save_path}")
    
    # Generate recent predictions
    print(f"\n🔮 Generating predictions for last 50 candles...")
    recent_df = df.tail(50)
    
    for model_name in ['random_forest', 'xgboost', 'logistic_regression']:
        if model_name not in models:
            continue
            
        predictions = engine.predict_signals(recent_df, model_name)
        
        signals = predictions['ml_signal'].sum()
        avg_prob = predictions['ml_probability'].mean()
        
        print(f"   {model_name:25s}: {signals:2d} signals, avg prob = {avg_prob:.3f}")
    
    print("\n✅ Training complete!")
    return timestamp


def test_predictions(symbol: str, model_timestamp: str | None = None, days: int = 30):
    """
    Test predictions with trained models
    
    Args:
        symbol: Trading pair
        model_timestamp: Timestamp of saved models (e.g., '20251226_123456')
        days: Number of recent days to predict on
    """
    print("=" * 80)
    print(f"🔮 AI Strategy Predictions: {symbol}")
    print("=" * 80)
    
    # Setup
    config = AIStrategyConfig()
    engine = ModelEngine(config)
    
    # Load models
    if model_timestamp:
        print(f"\n📦 Loading saved models: {model_timestamp}")
        engine.load_models(model_timestamp)
    else:
        print("\n⚠️  No model timestamp provided - using freshly trained models")
        # Train on the fly
        train_models(symbol, days=90)
    
    # Fetch recent data
    print(f"\n📊 Fetching {days} days of recent data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = get_historical_data(symbol, "1h", start_date, end_date)
    print(f"✅ Loaded {len(df)} candles")
    
    # Generate predictions
    print(f"\n🤖 Generating ML predictions...")
    predictions = engine.predict_signals(df, "random_forest")
    
    # Show last 10 signals
    signals_df = predictions[predictions['ml_signal'] == 1].tail(10)
    
    if len(signals_df) > 0:
        print(f"\n🎯 Last {len(signals_df)} Trading Signals:")
        print("-" * 80)
        
        for idx, row in signals_df.iterrows():
            print(f"\n📅 {idx}")
            print(f"   Price:       ${row['close']:,.2f}")
            print(f"   Prediction:  {'UP' if row['ml_prediction'] == 1 else 'DOWN'}")
            print(f"   Probability: {row['ml_probability']:.3f}")
            
            # Show key features
            if 'ema_fast_slow_diff' in row:
                print(f"   EMA Diff:    {row['ema_fast_slow_diff']:.2f}")
            if 'rsi_normalized' in row:
                print(f"   RSI Norm:    {row['rsi_normalized']:.3f}")
            if 'volatility' in row:
                print(f"   Volatility:  {row['volatility']:.5f}")
    else:
        print("\n⚠️  No trading signals generated in recent data")
    
    print("\n✅ Predictions complete!")


def test_llm_gate(symbol: str = "BTCUSDT"):
    """
    Test LLM decision gate with sample data
    
    Args:
        symbol: Trading pair
    """
    print("=" * 80)
    print(f"🧠 LLM Decision Gate Test: {symbol}")
    print("=" * 80)
    
    # Setup
    config = AIStrategyConfig()
    config.llm.provider = "mock"  # Use mock mode for demo
    
    llm = LLMGate(config)
    
    # Sample bullish scenario
    print("\n📊 Test Scenario 1: BULLISH SIGNAL")
    print("-" * 80)
    
    decision = llm.evaluate_candidate(
        symbol=symbol,
        current_price=50000.0,
        indicators={
            'ema_fast': 50500.0,
            'ema_slow': 49500.0,
            'rsi': 55.0,
            'atr': 1000.0,
            'macd': 150.0,
            'macd_signal': 100.0
        },
        ml_prediction={
            'prediction': 1,
            'probability': 0.75
        },
        recent_candles=[
            {'close': 49000, 'volume': 1000},
            {'close': 49500, 'volume': 1200},
            {'close': 50000, 'volume': 1500}
        ]
    )
    
    print(f"\n✅ Decision: {'APPROVE' if decision['approve_trade'] else 'REJECT'}")
    print(f"   Confidence: {decision['confidence']:.2f}")
    print(f"   Reasoning:  {decision['explanation']}")
    
    # Sample bearish scenario
    print("\n\n📊 Test Scenario 2: WEAK SIGNAL")
    print("-" * 80)
    
    decision = llm.evaluate_candidate(
        symbol=symbol,
        current_price=50000.0,
        indicators={
            'ema_fast': 49500.0,
            'ema_slow': 50500.0,
            'rsi': 75.0,  # Overbought
            'atr': 1000.0,
            'macd': -50.0,
            'macd_signal': 0.0
        },
        ml_prediction={
            'prediction': 1,
            'probability': 0.55  # Low confidence
        }
    )
    
    print(f"\n❌ Decision: {'APPROVE' if decision['approve_trade'] else 'REJECT'}")
    print(f"   Confidence: {decision['confidence']:.2f}")
    print(f"   Reasoning:  {decision['explanation']}")
    
    print("\n✅ LLM gate test complete!")


def full_pipeline_demo(symbol: str = "BTCUSDT", days: int = 60):
    """
    Demonstrate complete AI strategy pipeline
    
    Args:
        symbol: Trading pair
        days: Days of historical data
    """
    print("=" * 80)
    print(f"🚀 FULL AI STRATEGY PIPELINE DEMO")
    print("=" * 80)
    
    # Step 1: Train models
    print("\n" + "=" * 80)
    print("STEP 1: TRAIN ML MODELS")
    print("=" * 80)
    timestamp = train_models(symbol, days=days)
    
    # Step 2: Generate predictions
    print("\n\n" + "=" * 80)
    print("STEP 2: GENERATE PREDICTIONS")
    print("=" * 80)
    
    config = AIStrategyConfig()
    engine = ModelEngine(config)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = get_historical_data(symbol, "1h", start_date, end_date)
    
    predictions = engine.predict_signals(df.tail(100), "random_forest")
    
    # Get strongest signal
    strongest = predictions.nlargest(1, 'ml_probability')
    
    if len(strongest) > 0:
        signal = strongest.iloc[0]
        
        # Step 3: LLM evaluation
        print("\n\n" + "=" * 80)
        print("STEP 3: LLM DECISION GATE")
        print("=" * 80)
        
        config.llm.provider = "mock"
        llm = LLMGate(config)
        
        decision = llm.evaluate_candidate(
            symbol=symbol,
            current_price=signal['close'],
            indicators={
                'ema_fast': signal.get('ema_fast', 0),
                'ema_slow': signal.get('ema_slow', 0),
                'rsi': signal.get('rsi', 50),
                'atr': signal.get('atr', 0),
            },
            ml_prediction={
                'prediction': int(signal['ml_prediction']),
                'probability': float(signal['ml_probability'])
            }
        )
        
        print(f"\n📊 Candidate Trade:")
        print(f"   Symbol:      {symbol}")
        print(f"   Price:       ${signal['close']:,.2f}")
        print(f"   ML Prob:     {signal['ml_probability']:.3f}")
        
        print(f"\n🧠 LLM Decision:")
        print(f"   Approve:     {'✅ YES' if decision['approve_trade'] else '❌ NO'}")
        print(f"   Confidence:  {decision['confidence']:.2f}")
        print(f"   Reasoning:   {decision['explanation']}")
    
    print("\n\n" + "=" * 80)
    print("✅ FULL PIPELINE DEMO COMPLETE!")
    print("=" * 80)
    print(f"\n💡 Next steps:")
    print(f"   1. Integrate with backtesting engine")
    print(f"   2. Add risk management layer")
    print(f"   3. Connect to WEEX API for live trading")


def main():
    parser = argparse.ArgumentParser(
        description="AI Strategy Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python run_ai_demo.py --train --symbol BTCUSDT --days 90
  
  # Test predictions
  python run_ai_demo.py --predict --symbol ETHUSDT
  
  # Test LLM gate
  python run_ai_demo.py --test-llm
  
  # Full pipeline demo
  python run_ai_demo.py --full-demo --symbol BTCUSDT
        """
    )
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=90,
                        help='Days of historical data (default: 90)')
    parser.add_argument('--interval', type=str, default='1h',
                        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                        help='Candle interval (default: 1h)')
    
    # Action flags
    parser.add_argument('--train', action='store_true',
                        help='Train ML models')
    parser.add_argument('--predict', action='store_true',
                        help='Generate predictions')
    parser.add_argument('--test-llm', action='store_true',
                        help='Test LLM decision gate')
    parser.add_argument('--full-demo', action='store_true',
                        help='Run full pipeline demo')
    
    parser.add_argument('--model-timestamp', type=str,
                        help='Load models from timestamp (e.g., 20251226_123456)')
    
    args = parser.parse_args()
    
    try:
        if args.train:
            train_models(args.symbol, args.days, args.interval)
        
        elif args.predict:
            test_predictions(args.symbol, args.model_timestamp, days=30)
        
        elif args.test_llm:
            test_llm_gate(args.symbol)
        
        elif args.full_demo:
            full_pipeline_demo(args.symbol, args.days)
        
        else:
            # Default: show help
            parser.print_help()
            print("\n💡 Tip: Try 'python run_ai_demo.py --full-demo' to see everything in action!")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
