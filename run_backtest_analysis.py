#!/usr/bin/env python3
"""
AI Strategy Backtest Runner
============================

Runs complete AI strategy backtests and generates comprehensive analysis.

Usage:
    python run_backtest_analysis.py --symbol BTCUSDT --days 180
    python run_backtest_analysis.py --symbol ETHUSDT --compare-models
    python run_backtest_analysis.py --symbol SOLUSDT --optimize
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_strategy.config import AIStrategyConfig
from ai_strategy.ai_backtester import AIBacktester


def fetch_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """
    Fetch and prepare historical data with indicators
    
    Args:
        symbol: Trading symbol
        days: Number of days of history
    
    Returns:
        DataFrame with OHLCV and indicators
    """
    print(f"📊 Fetching {days} days of data for {symbol}...")
    
    # Import with proper path handling
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "data_fetcher",
        str(project_root / "backtesting" / "data_fetcher.py")
    )
    if not spec or not spec.loader:
        raise ImportError("Could not load data_fetcher module")
    
    data_fetcher = importlib.util.module_from_spec(spec)
    sys.modules["data_fetcher"] = data_fetcher
    spec.loader.exec_module(data_fetcher)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = data_fetcher.get_historical_data(symbol, "1h", start_date, end_date)
    
    # Add indicators
    spec = importlib.util.spec_from_file_location(
        "indicators",
        str(project_root / "backtesting" / "indicators.py")
    )
    if not spec or not spec.loader:
        raise ImportError("Could not load indicators module")
    
    indicators = importlib.util.module_from_spec(spec)
    sys.modules["indicators"] = indicators
    spec.loader.exec_module(indicators)
    
    print("📈 Calculating indicators...")
    df = indicators.calculate_ema(df, 20)
    df = indicators.calculate_ema(df, 50)
    df = indicators.calculate_ema(df, 200)
    df.rename(columns={'ema': 'ema_fast'}, inplace=True)
    df.rename(columns={'ema_50': 'ema_slow'}, inplace=True)
    
    df = indicators.calculate_atr(df, 14)
    df = indicators.calculate_rsi(df, 14)
    df = indicators.calculate_macd(df, 12, 26, 9)
    df = indicators.calculate_bollinger_bands(df, 20, 2.0)
    
    # Remove NaN rows from indicator calculation
    df = df.dropna()
    
    print(f"✅ Loaded {len(df)} candles with indicators")
    return df


def run_single_backtest(
    symbol: str,
    days: int = 180,
    model_name: str = "random_forest",
    use_llm_gate: bool = True
):
    """Run a single backtest with specified parameters"""
    print("\n" + "="*80)
    print(f"🚀 AI STRATEGY BACKTEST - {symbol}")
    print("="*80)
    
    # Fetch data
    df = fetch_data(symbol, days)
    
    # Configure strategy
    config = AIStrategyConfig()
    config.llm.provider = "mock"  # Use mock mode for fast testing
    config.llm.use_llm_gate = use_llm_gate
    
    # Run backtest
    backtester = AIBacktester(config)
    results = backtester.run_backtest(
        df=df,
        symbol=symbol,
        model_name=model_name,
        use_llm_gate=use_llm_gate,
        train_size=0.7
    )
    
    # Print results
    backtester.print_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{symbol}_{model_name}_{timestamp}.json"
    backtester.save_results(results, filename)
    
    return results, backtester


def compare_models(symbol: str, days: int = 180):
    """Compare performance of different ML models"""
    print("\n" + "="*80)
    print(f"🔬 MODEL COMPARISON - {symbol}")
    print("="*80)
    
    # Fetch data once
    df = fetch_data(symbol, days)
    
    models = ["random_forest", "xgboost", "logistic_regression"]
    results_list = []
    
    for model_name in models:
        print(f"\n{'─'*80}")
        print(f"Testing {model_name.upper()}")
        print(f"{'─'*80}")
        
        config = AIStrategyConfig()
        config.llm.provider = "mock"
        config.llm.use_llm_gate = True
        
        backtester = AIBacktester(config)
        
        try:
            results = backtester.run_backtest(
                df=df,
                symbol=symbol,
                model_name=model_name,
                use_llm_gate=True,
                train_size=0.7
            )
            results_list.append((model_name, results, backtester))
            
            # Print summary
            metrics = results['metrics']
            print(f"\n📊 {model_name.upper()} Summary:")
            print(f"  Total Return: {metrics['total_return']:.2%}")
            print(f"  Win Rate: {metrics['win_rate']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  Total Trades: {metrics['total_trades']}")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
    
    # Comparison summary
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Return':>12} {'Win Rate':>12} {'Sharpe':>12} {'Trades':>10}")
    print("─"*80)
    
    for model_name, results, _ in results_list:
        metrics = results['metrics']
        print(f"{model_name:<20} {metrics['total_return']:>11.2%} {metrics['win_rate']:>11.2%} {metrics['sharpe_ratio']:>11.2f} {metrics['total_trades']:>10}")
    
    # Best model
    if results_list:
        best = max(results_list, key=lambda x: x[1]['metrics']['total_return'])
        print(f"\n🏆 Best Model: {best[0].upper()} (Return: {best[1]['metrics']['total_return']:.2%})")
    
    return results_list


def analyze_feature_importance(symbol: str, days: int = 180):
    """Analyze which features are most important for predictions"""
    print("\n" + "="*80)
    print(f"🔍 FEATURE IMPORTANCE ANALYSIS - {symbol}")
    print("="*80)
    
    df = fetch_data(symbol, days)
    
    config = AIStrategyConfig()
    backtester = AIBacktester(config)
    
    # Train models
    train_df = df.iloc[:int(len(df)*0.7)]
    backtester.train_models(train_df)
    
    # Get feature importance
    print("\n📊 Top 15 Most Important Features (Random Forest):")
    print("─"*80)
    
    importance = backtester.model_engine.get_feature_importance("random_forest", top_n=15)
    if importance is not None and len(importance) > 0:
        for i, row in enumerate(importance, 1):
            feature = row[0] if hasattr(row, '__getitem__') else str(row)
            score = row[1] if hasattr(row, '__getitem__') and len(row) > 1 else 0
            print(f"{i:2d}. {feature:<30s} {'▓' * int(score * 50)} {score:.4f}")
    
    # Model performance comparison
    print("\n📈 Model Training Performance:")
    print("─"*80)
    
    for model_name, metrics in backtester.model_engine.training_metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  F1 Score:        {metrics['f1']:.4f}")
        print(f"  AUC:             {metrics['auc']:.4f}")


def test_llm_impact(symbol: str, days: int = 180):
    """Test impact of LLM decision gate on performance"""
    print("\n" + "="*80)
    print(f"🧠 LLM GATE IMPACT ANALYSIS - {symbol}")
    print("="*80)
    
    df = fetch_data(symbol, days)
    
    # Test with and without LLM gate
    configs = [
        ("Without LLM Gate", False),
        ("With LLM Gate", True),
    ]
    
    results_dict = {}
    
    for name, use_llm in configs:
        print(f"\n{'─'*80}")
        print(f"Testing: {name}")
        print(f"{'─'*80}")
        
        config = AIStrategyConfig()
        config.llm.provider = "mock"
        config.llm.use_llm_gate = use_llm
        
        backtester = AIBacktester(config)
        results = backtester.run_backtest(
            df=df,
            symbol=symbol,
            model_name="random_forest",
            use_llm_gate=use_llm,
            train_size=0.7
        )
        
        results_dict[name] = results
        
        metrics = results['metrics']
        print(f"\n📊 Summary:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 LLM GATE IMPACT COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Without LLM':>15} {'With LLM':>15} {'Difference':>15}")
    print("─"*80)
    
    no_llm = results_dict["Without LLM Gate"]['metrics']
    with_llm = results_dict["With LLM Gate"]['metrics']
    
    comparisons = [
        ("Total Return", no_llm['total_return'], with_llm['total_return'], "%"),
        ("Win Rate", no_llm['win_rate'], with_llm['win_rate'], "%"),
        ("Total Trades", no_llm['total_trades'], with_llm['total_trades'], ""),
        ("Sharpe Ratio", no_llm['sharpe_ratio'], with_llm['sharpe_ratio'], ""),
        ("Max Drawdown", no_llm['max_drawdown'], with_llm['max_drawdown'], "%"),
    ]
    
    for metric, val1, val2, unit in comparisons:
        if unit == "%":
            diff = val2 - val1
            print(f"{metric:<25} {val1:>14.2%} {val2:>14.2%} {diff:>+14.2%}")
        else:
            diff = val2 - val1
            print(f"{metric:<25} {val1:>14.2f} {val2:>14.2f} {diff:>+14.2f}")
    
    return results_dict


def optimize_parameters(symbol: str, days: int = 180):
    """Test different parameter combinations to find optimal settings"""
    print("\n" + "="*80)
    print(f"⚙️  PARAMETER OPTIMIZATION - {symbol}")
    print("="*80)
    
    df = fetch_data(symbol, days)
    
    # Parameter grid
    param_grid = {
        'prediction_threshold': [0.55, 0.60, 0.65],
        'max_position_size_percent': [0.01, 0.02, 0.03],
        'stop_loss_atr_multiplier': [1.0, 1.5, 2.0],
    }
    
    best_return = -float('inf')
    best_params = None
    results_list = []
    
    total_combinations = (
        len(param_grid['prediction_threshold']) *
        len(param_grid['max_position_size_percent']) *
        len(param_grid['stop_loss_atr_multiplier'])
    )
    
    print(f"\n🔧 Testing {total_combinations} parameter combinations...")
    
    combination = 0
    for pred_threshold in param_grid['prediction_threshold']:
        for pos_size in param_grid['max_position_size_percent']:
            for sl_mult in param_grid['stop_loss_atr_multiplier']:
                combination += 1
                
                config = AIStrategyConfig()
                config.llm.provider = "mock"
                config.model.prediction_threshold = pred_threshold
                config.risk.max_position_size_percent = pos_size
                config.risk.stop_loss_atr_multiplier = sl_mult
                config.risk.take_profit_atr_multiplier = sl_mult * 2  # 2:1 R/R
                
                try:
                    backtester = AIBacktester(config)
                    results = backtester.run_backtest(
                        df=df,
                        symbol=symbol,
                        model_name="random_forest",
                        use_llm_gate=True,
                        train_size=0.7
                    )
                    
                    total_return = results['metrics']['total_return']
                    win_rate = results['metrics']['win_rate']
                    trades = results['metrics']['total_trades']
                    
                    results_list.append({
                        'pred_threshold': pred_threshold,
                        'pos_size': pos_size,
                        'sl_mult': sl_mult,
                        'return': total_return,
                        'win_rate': win_rate,
                        'trades': trades,
                    })
                    
                    if total_return > best_return and trades >= 10:  # Minimum 10 trades
                        best_return = total_return
                        best_params = {
                            'prediction_threshold': pred_threshold,
                            'max_position_size': pos_size,
                            'stop_loss_multiplier': sl_mult,
                        }
                    
                    print(f"  [{combination}/{total_combinations}] "
                          f"Pred={pred_threshold:.2f}, Size={pos_size:.2%}, SL={sl_mult:.1f} → "
                          f"Return={total_return:+.2%}, WR={win_rate:.1%}, Trades={trades}")
                    
                except Exception as e:
                    print(f"  [{combination}/{total_combinations}] Error: {e}")
    
    # Results summary
    print("\n" + "="*80)
    print("🏆 OPTIMIZATION RESULTS")
    print("="*80)
    
    if best_params:
        print(f"\n✅ Best Parameters (Return: {best_return:.2%}):")
        print(f"  Prediction Threshold: {best_params['prediction_threshold']:.2f}")
        print(f"  Max Position Size: {best_params['max_position_size']:.2%}")
        print(f"  Stop Loss Multiplier: {best_params['stop_loss_multiplier']:.1f}x ATR")
    
    # Top 5 configurations
    print(f"\n📊 Top 5 Configurations by Return:")
    print("─"*80)
    
    sorted_results = sorted(results_list, key=lambda x: x['return'], reverse=True)[:5]
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. Pred={result['pred_threshold']:.2f}, Size={result['pos_size']:.2%}, "
              f"SL={result['sl_mult']:.1f}x → Return={result['return']:+.2%}, "
              f"WR={result['win_rate']:.1%}, Trades={result['trades']}")
    
    return best_params, results_list


def main():
    parser = argparse.ArgumentParser(
        description="AI Strategy Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=180,
                        help='Days of historical data (default: 180)')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'logistic_regression'],
                        help='ML model to use (default: random_forest)')
    
    # Analysis modes
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare all ML models')
    parser.add_argument('--feature-importance', action='store_true',
                        help='Analyze feature importance')
    parser.add_argument('--test-llm-impact', action='store_true',
                        help='Test LLM gate impact')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize parameters')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM gate')
    
    args = parser.parse_args()
    
    try:
        if args.compare_models:
            compare_models(args.symbol, args.days)
        
        elif args.feature_importance:
            analyze_feature_importance(args.symbol, args.days)
        
        elif args.test_llm_impact:
            test_llm_impact(args.symbol, args.days)
        
        elif args.optimize:
            optimize_parameters(args.symbol, args.days)
        
        else:
            # Single backtest
            run_single_backtest(
                symbol=args.symbol,
                days=args.days,
                model_name=args.model,
                use_llm_gate=not args.no_llm
            )
        
        print("\n✅ Analysis complete!")
        
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
