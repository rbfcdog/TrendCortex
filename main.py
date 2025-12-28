"""
TrendCortex Main Entry Point

Orchestrates the complete trading system:
1. Data fetching
2. Signal generation
3. AI decision making
4. Risk validation
5. Trade execution
6. Logging and monitoring
"""

import asyncio
import argparse
import signal as sys_signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from trendcortex.api_client import WEEXAPIClient
from trendcortex.config import Config
from trendcortex.data_manager import DataManager
from trendcortex.execution import TradeExecutor
from trendcortex.logger import setup_logging, get_logger
from trendcortex.model_integration import HybridDecisionEngine
from trendcortex.risk_controller import RiskController
from trendcortex.signal_engine import SignalEngine


class TrendCortexBot:
    """
    Main trading bot orchestrator.
    
    Runs the complete trading loop with proper error handling
    and graceful shutdown.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize TrendCortex bot.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config.load(config_path)
        self.config.validate_trading_rules()
        
        # Setup logging
        self.logger = setup_logging(
            log_dir=self.config.logging.log_directory,
            level=self.config.logging.level,
            console_output=self.config.logging.console_output,
            json_format=self.config.logging.json_format,
        )
        
        self.logger.info("=" * 60)
        self.logger.info("TrendCortex v1.0.0 - Starting...")
        self.logger.info("=" * 60)
        
        # Initialize components
        self.api_client: Optional[WEEXAPIClient] = None
        self.data_manager: Optional[DataManager] = None
        self.signal_engine: Optional[SignalEngine] = None
        self.decision_engine: Optional[HybridDecisionEngine] = None
        self.risk_controller: Optional[RiskController] = None
        self.executor: Optional[TradeExecutor] = None
        
        # Runtime state
        self.running = False
        self.shutdown_requested = False
        
        # Performance tracking
        self.total_signals = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
    
    async def initialize(self) -> None:
        """Initialize all system components."""
        self.logger.info("Initializing components...")
        
        # API client
        self.api_client = WEEXAPIClient(self.config)
        await self.api_client.connect()
        
        # Test API connection
        try:
            server_time = await self.api_client.get_server_time()
            self.logger.info(f"Connected to WEEX API - Server time: {server_time.get('iso')}")
        except Exception as e:
            self.logger.error(f"Failed to connect to WEEX API: {e}")
            raise
        
        # Data manager
        self.data_manager = DataManager(self.api_client, self.config)
        
        # Signal engine
        self.signal_engine = SignalEngine(self.config)
        
        # Decision engine (ML + LLM)
        self.decision_engine = HybridDecisionEngine(self.config)
        
        # Risk controller
        self.risk_controller = RiskController(self.config)
        
        # Trade executor
        self.executor = TradeExecutor(self.api_client, self.config)
        
        if self.config.execution.dry_run:
            self.logger.warning("🔶 RUNNING IN DRY-RUN MODE - No real trades will be executed")
        else:
            self.logger.info("🔴 RUNNING IN LIVE MODE - Real money at risk!")
        
        self.logger.info("✅ All components initialized successfully")
    
    async def run(self) -> None:
        """Main trading loop."""
        self.running = True
        
        while self.running and not self.shutdown_requested:
            try:
                loop_start = datetime.now()
                
                # Run trading cycle for each enabled symbol
                for symbol in self.config.trading.symbols:
                    if self.shutdown_requested:
                        break
                    
                    await self._process_symbol(symbol)
                
                # Log performance metrics periodically
                if self.total_signals > 0 and self.total_signals % 10 == 0:
                    await self._log_performance()
                
                # Calculate sleep time to maintain loop interval
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, self.config.system.loop_interval_seconds - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                self.shutdown_requested = True
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                
                if self.config.system.auto_restart_on_error:
                    self.logger.info("Auto-restart enabled, continuing...")
                    await asyncio.sleep(5)
                else:
                    self.logger.error("Auto-restart disabled, stopping bot")
                    break
        
        await self.shutdown()
    
    async def _process_symbol(self, symbol: str) -> None:
        """
        Process trading logic for a single symbol.
        
        Args:
            symbol: Trading pair symbol
        """
        try:
            # Step 1: Fetch market data
            timeframes = self.config.trading.timeframes
            market_data = await self.data_manager.get_multi_timeframe_data(
                symbol=symbol,
                timeframes=list(timeframes.values()),
                limit=self.config.data.warmup_periods,
            )
            
            primary_tf = timeframes["primary"]
            df = market_data.get(primary_tf)
            
            if df is None or df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return
            
            # Get ticker data for context
            ticker = await self.data_manager.get_ticker_data(symbol)
            if not ticker:
                self.logger.warning(f"No ticker data for {symbol}")
                return
            
            # Step 2: Generate signals
            signals = self.signal_engine.generate_signals(symbol, df)
            
            if not signals:
                self.logger.debug(f"No signals generated for {symbol}")
                return
            
            self.logger.info(f"📊 Generated {len(signals)} signal(s) for {symbol}")
            self.total_signals += len(signals)
            
            # Step 3: Process each signal
            for signal in signals:
                if self.shutdown_requested:
                    break
                
                # Log signal
                self.logger.log_signal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
                    direction=signal.direction.value,
                    confidence=signal.confidence,
                    price=signal.price,
                    indicators=signal.indicators,
                    metadata=signal.metadata,
                )
                
                # Step 4: AI decision
                decision = await self.decision_engine.evaluate_signal(
                    signal=signal,
                    market_data=df,
                    market_context=ticker,
                )
                
                # Log decision
                self.logger.log_decision(
                    symbol=symbol,
                    signal_data={
                        "type": signal.signal_type.value,
                        "direction": signal.direction.value,
                        "confidence": signal.confidence,
                    },
                    ml_score=decision.score,
                    llm_decision=None,
                    final_decision="approve" if decision.approve_trade else "reject",
                    approve_trade=decision.approve_trade,
                    confidence=decision.confidence,
                    explanation=decision.explanation,
                )
                
                if not decision.approve_trade:
                    self.logger.info(f"❌ Trade rejected by AI: {decision.explanation}")
                    continue
                
                # Step 5: Risk validation
                account_balance = await self._get_account_balance()
                open_positions = await self._get_open_positions()
                current_volatility = signal.indicators.get("atr_percent", 0)
                
                risk_assessment = await self.risk_controller.validate_trade(
                    signal=signal,
                    account_balance=account_balance,
                    open_positions=open_positions,
                    current_volatility=current_volatility,
                )
                
                if not risk_assessment.approved:
                    self.logger.info(f"❌ Trade rejected by risk: {risk_assessment.reason}")
                    continue
                
                if risk_assessment.warnings:
                    for warning in risk_assessment.warnings:
                        self.logger.warning(f"⚠️ Risk warning: {warning}")
                
                # Step 6: Execute trade
                self.logger.info(
                    f"✅ Executing {signal.direction.value} trade for {symbol} "
                    f"(Size: {risk_assessment.suggested_size:.4f}, "
                    f"Leverage: {risk_assessment.suggested_leverage}x)"
                )
                
                contract_specs = await self.data_manager.get_contract_specs(symbol)
                
                execution_result = await self.executor.execute_signal(
                    signal=signal,
                    risk_assessment=risk_assessment,
                    contract_specs=contract_specs,
                )
                
                self.total_trades += 1
                
                if execution_result.success:
                    self.successful_trades += 1
                    self.logger.info(
                        f"✅ Trade executed successfully: {execution_result.order_id} "
                        f"(Filled: {execution_result.filled_size} @ {execution_result.filled_price})"
                    )
                    
                    # Record for cooldown
                    self.risk_controller.record_trade(symbol)
                else:
                    self.failed_trades += 1
                    self.logger.error(
                        f"❌ Trade execution failed: {execution_result.error_message}"
                    )
        
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}", exc_info=True)
    
    async def _get_account_balance(self) -> float:
        """Get available account balance."""
        try:
            balances = await self.api_client.get_account_balance()
            
            # Find USDT balance
            for balance in balances:
                if balance.get("coinName") == "USDT":
                    return float(balance.get("available", 0))
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            return 0.0
    
    async def _get_open_positions(self) -> list:
        """Get list of open positions."""
        try:
            positions = await self.api_client.get_positions()
            return positions or []
        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            return []
    
    async def _log_performance(self) -> None:
        """Log performance metrics."""
        account_balance = await self._get_account_balance()
        open_positions = await self._get_open_positions()
        
        win_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        self.logger.log_performance(
            total_pnl=0.0,  # TODO: Calculate from trade history
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            win_rate=win_rate,
            total_trades=self.total_trades,
            open_positions=len(open_positions),
            account_balance=account_balance,
            metadata={
                "total_signals": self.total_signals,
                "successful_trades": self.successful_trades,
                "failed_trades": self.failed_trades,
            }
        )
    
    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.info("Shutting down TrendCortex...")
        
        self.running = False
        
        # Close API connection
        if self.api_client:
            await self.api_client.close()
        
        # Final performance log
        await self._log_performance()
        
        self.logger.info("=" * 60)
        self.logger.info(f"Session Summary:")
        self.logger.info(f"  Total Signals: {self.total_signals}")
        self.logger.info(f"  Total Trades: {self.total_trades}")
        self.logger.info(f"  Successful: {self.successful_trades}")
        self.logger.info(f"  Failed: {self.failed_trades}")
        self.logger.info("=" * 60)
        self.logger.info("TrendCortex stopped. Goodbye! 👋")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TrendCortex Crypto Trading Bot")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no real trades)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode",
    )
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = TrendCortexBot(config_path=args.config)
    
    # Override dry-run setting if specified
    if args.dry_run:
        bot.config.execution.dry_run = True
    elif args.live:
        bot.config.execution.dry_run = False
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received...")
        bot.shutdown_requested = True
    
    sys_signal.signal(sys_signal.SIGINT, signal_handler)
    sys_signal.signal(sys_signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and run
        await bot.initialize()
        await bot.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
