"""
Backtest Runner Script

This is the main CLI interface for running backtests on multiple symbols
and timeframes. It handles data fetching, backtest execution, and results
aggregation.

Usage:
    python run_backtest.py
    python run_backtest.py --symbols BTCUSDT ETHUSDT --interval 1h --days 90
    python run_backtest.py --all-symbols --interval 15m --days 30
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import List
import logging

from config import (
    APPROVED_SYMBOLS,
    DEFAULT_INTERVALS,
    DEFAULT_DAYS_BACK,
    validate_symbol,
    validate_interval,
    get_results_path,
    RESULTS_DIR,
)
from data_fetcher import get_historical_data
from backtester import Backtester, EMACrossoverStrategy
from indicators import add_all_indicators

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run cryptocurrency backtests on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest BTC on 1h timeframe for last 90 days
  python run_backtest.py --symbols BTCUSDT --interval 1h --days 90
  
  # Backtest multiple pairs on 15m timeframe
  python run_backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 15m --days 30
  
  # Backtest all approved symbols
  python run_backtest.py --all-symbols --interval 1h --days 180
  
  # Use custom date range
  python run_backtest.py --symbols BTCUSDT --interval 4h --start 2024-01-01 --end 2024-12-31
  
  # Save detailed results
  python run_backtest.py --symbols BTCUSDT --interval 1h --days 90 --save-results
        """
    )
    
    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        '--symbols',
        nargs='+',
        help='List of symbols to backtest (e.g., BTCUSDT ETHUSDT)'
    )
    symbol_group.add_argument(
        '--all-symbols',
        action='store_true',
        help='Backtest all approved symbols'
    )
    
    # Timeframe
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        help='Timeframe interval (e.g., 1h, 15m, 4h). Default: 1h'
    )
    
    # Date range options
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        '--days',
        type=int,
        default=DEFAULT_DAYS_BACK,
        help=f'Number of days to backtest (default: {DEFAULT_DAYS_BACK})'
    )
    date_group.add_argument(
        '--date-range',
        type=str,
        nargs=2,
        metavar=('START', 'END'),
        help='Custom date range (format: YYYY-MM-DD YYYY-MM-DD)'
    )
    
    # Output options
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results to CSV files'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force refresh data from API (ignore cache)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Strategy parameters
    parser.add_argument(
        '--fast-ema',
        type=int,
        default=20,
        help='Fast EMA period (default: 20)'
    )
    parser.add_argument(
        '--slow-ema',
        type=int,
        default=50,
        help='Slow EMA period (default: 50)'
    )
    parser.add_argument(
        '--atr-period',
        type=int,
        default=14,
        help='ATR period (default: 14)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """
    Validate parsed arguments.
    
    Args:
        args: Parsed arguments namespace
    
    Raises:
        ValueError: If arguments are invalid
    """
    # Validate symbols
    if args.symbols:
        for symbol in args.symbols:
            if not validate_symbol(symbol):
                raise ValueError(
                    f"Invalid symbol: {symbol}. "
                    f"Must be one of: {', '.join(APPROVED_SYMBOLS)}"
                )
    
    # Validate interval
    if not validate_interval(args.interval):
        raise ValueError(f"Invalid interval: {args.interval}")
    
    # Validate date range if provided
    if args.date_range:
        try:
            start_date = datetime.strptime(args.date_range[0], '%Y-%m-%d')
            end_date = datetime.strptime(args.date_range[1], '%Y-%m-%d')
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            if end_date > datetime.now():
                raise ValueError("End date cannot be in the future")
                
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")


def get_date_range(args):
    """
    Get start and end dates from arguments.
    
    Args:
        args: Parsed arguments
    
    Returns:
        Tuple of (start_date, end_date)
    """
    if args.date_range:
        start_date = datetime.strptime(args.date_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(args.date_range[1], '%Y-%m-%d')
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    
    return start_date, end_date


def run_single_backtest(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    strategy_params: dict,
    use_cache: bool = True
) -> dict:
    """
    Run backtest for a single symbol.
    
    Args:
        symbol: Trading pair symbol
        interval: Timeframe interval
        start_date: Start date
        end_date: End date
        strategy_params: Strategy parameters
        use_cache: Whether to use cached data
    
    Returns:
        Dictionary with backtest results
    """
    logger.info("=" * 80)
    logger.info(f"Running backtest: {symbol} {interval}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info("=" * 80)
    
    try:
        # 1. Fetch historical data
        logger.info("Fetching historical data...")
        df = get_historical_data(
            symbol,
            interval,
            start_date,
            end_date,
            use_cache=use_cache,
            force_refresh=not use_cache
        )
        
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        logger.info(f"Loaded {len(df)} candles")
        
        # 2. Create strategy
        strategy = EMACrossoverStrategy(params=strategy_params)
        
        # 3. Run backtest
        backtester = Backtester(df, strategy)
        results_df = backtester.run()
        
        # 4. Print summary
        backtester.print_summary()
        
        # 5. Prepare results summary
        if results_df.empty:
            summary = {
                'symbol': symbol,
                'interval': interval,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'final_capital': backtester.initial_capital,
            }
        else:
            total_trades = len(results_df)
            winning_trades = len(results_df[results_df['pnl'] > 0])
            losing_trades = len(results_df[results_df['pnl'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = results_df['pnl'].sum()
            final_capital = backtester.capital
            total_return = ((final_capital - backtester.initial_capital) / backtester.initial_capital) * 100
            
            summary = {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date,
                'total_candles': len(df),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return_pct': total_return,
                'initial_capital': backtester.initial_capital,
                'final_capital': final_capital,
                'avg_trade_pnl': results_df['pnl'].mean() if not results_df.empty else 0,
                'best_trade': results_df['pnl'].max() if not results_df.empty else 0,
                'worst_trade': results_df['pnl'].min() if not results_df.empty else 0,
                'results_df': results_df,
                'equity_curve': backtester.equity_curve,
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(summary: dict, save_trades: bool = True):
    """
    Save backtest results to files.
    
    Args:
        summary: Results summary dictionary
        save_trades: Whether to save individual trades
    """
    symbol = summary['symbol']
    interval = summary['interval']
    
    # Save trade results
    if save_trades and 'results_df' in summary and not summary['results_df'].empty:
        results_path = get_results_path(symbol, interval)
        summary['results_df'].to_csv(results_path, index=False)
        logger.info(f"Saved trade results to: {results_path}")
    
    # Save equity curve
    if 'equity_curve' in summary and summary['equity_curve']:
        equity_path = RESULTS_DIR / f"equity_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        equity_df = pd.DataFrame({
            'bar_index': range(len(summary['equity_curve'])),
            'equity': summary['equity_curve']
        })
        equity_df.to_csv(equity_path, index=False)
        logger.info(f"Saved equity curve to: {equity_path}")


def print_overall_summary(results: List[dict]):
    """
    Print summary across all backtests.
    
    Args:
        results: List of backtest result dictionaries
    """
    if not results:
        print("\nNo successful backtests to summarize")
        return
    
    print("\n" + "=" * 100)
    print("OVERALL BACKTEST SUMMARY")
    print("=" * 100)
    
    # Create summary table
    summary_data = []
    for r in results:
        if r is not None:
            summary_data.append({
                'Symbol': r['symbol'],
                'Interval': r['interval'],
                'Trades': r['total_trades'],
                'Win Rate': f"{r['win_rate']:.1f}%",
                'Total P&L': f"${r['total_pnl']:+,.2f}",
                'Return': f"{r['total_return_pct']:+.2f}%",
                'Final Capital': f"${r['final_capital']:,.2f}",
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + df.to_string(index=False))
        
        # Overall statistics
        total_trades = sum(r['total_trades'] for r in results if r)
        total_pnl = sum(r['total_pnl'] for r in results if r)
        avg_win_rate = sum(r['win_rate'] for r in results if r) / len(results)
        
        print("\n" + "=" * 100)
        print(f"Total Trades Across All Symbols: {total_trades}")
        print(f"Combined P&L: ${total_pnl:+,.2f}")
        print(f"Average Win Rate: {avg_win_rate:.1f}%")
        print("=" * 100)


def main():
    """
    Main execution function.
    """
    # Parse and validate arguments
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get symbols to backtest
    if args.all_symbols:
        symbols = APPROVED_SYMBOLS
        logger.info(f"Backtesting all {len(symbols)} approved symbols")
    else:
        symbols = args.symbols
        logger.info(f"Backtesting {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Get date range
    start_date, end_date = get_date_range(args)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Strategy parameters
    strategy_params = {
        'fast_ema': args.fast_ema,
        'slow_ema': args.slow_ema,
        'atr_period': args.atr_period,
        'ema_fast': args.fast_ema,  # For indicators module
        'ema_slow': args.slow_ema,
    }
    logger.info(f"Strategy parameters: {strategy_params}")
    
    # Run backtests
    results = []
    
    for symbol in symbols:
        result = run_single_backtest(
            symbol=symbol,
            interval=args.interval,
            start_date=start_date,
            end_date=end_date,
            strategy_params=strategy_params,
            use_cache=not args.no_cache
        )
        
        if result:
            results.append(result)
            
            # Save results if requested
            if args.save_results:
                save_results(result)
        
        print("\n")  # Spacing between backtests
    
    # Print overall summary
    print_overall_summary(results)
    
    logger.info(f"\nBacktesting complete! Results saved to: {RESULTS_DIR}")
    
    return 0


if __name__ == "__main__":
    exit(main())
