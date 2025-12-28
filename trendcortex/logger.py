"""
Structured Logging Module

Provides JSON-formatted logging for signals, decisions, executions, and system events.
Supports both file and console output with automatic log rotation.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


class TrendCortexLogger:
    """
    Structured logger for TrendCortex trading system.
    
    Provides separate log streams for:
    - Signals (technical analysis signals)
    - Decisions (ML/LLM decision outputs)
    - Executions (trade execution results)
    - System (general application logs)
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        level: str = "INFO",
        console_output: bool = True,
        json_format: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.level = getattr(logging, level.upper())
        self.console_output = console_output
        self.json_format = json_format
        
        # Create log directories
        self._create_log_directories()
        
        # Initialize loggers
        self.signal_logger = self._setup_logger("signals", "signals")
        self.decision_logger = self._setup_logger("decisions", "decisions")
        self.execution_logger = self._setup_logger("executions", "executions")
        self.system_logger = self._setup_logger("system", "")
    
    def _create_log_directories(self) -> None:
        """Create log directory structure."""
        directories = [
            self.log_dir,
            self.log_dir / "signals",
            self.log_dir / "decisions",
            self.log_dir / "executions",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self, name: str, subdir: str) -> logging.Logger:
        """
        Setup a logger with file and console handlers.
        
        Args:
            name: Logger name
            subdir: Subdirectory for log files (empty for root log dir)
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"trendcortex.{name}")
        logger.setLevel(self.level)
        logger.handlers.clear()  # Clear existing handlers
        
        # File handler
        log_path = self.log_dir
        if subdir:
            log_path = log_path / subdir
        
        today = datetime.now().strftime("%Y%m%d")
        file_path = log_path / f"{name}_{today}.json"
        
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(self.level)
        
        if self.json_format:
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                timestamp=True,
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        direction: str,
        confidence: float,
        price: float,
        indicators: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a trading signal.
        
        Args:
            symbol: Trading pair symbol
            signal_type: Type of signal (e.g., "ema_cross", "rsi_oversold")
            direction: Signal direction ("long" or "short")
            confidence: Signal confidence score (0-1)
            price: Current price when signal generated
            indicators: Dictionary of indicator values
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": direction,
            "confidence": confidence,
            "price": price,
            "indicators": indicators,
        }
        
        if metadata:
            log_entry["metadata"] = metadata
        
        self.signal_logger.info(json.dumps(log_entry))
    
    def log_decision(
        self,
        symbol: str,
        signal_data: Dict[str, Any],
        ml_score: Optional[float],
        llm_decision: Optional[Dict[str, Any]],
        final_decision: str,
        approve_trade: bool,
        confidence: float,
        explanation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an AI decision (ML + LLM).
        
        Args:
            symbol: Trading pair symbol
            signal_data: Original signal data
            ml_score: ML model confidence score
            llm_decision: LLM decision output
            final_decision: Final decision ("approve", "reject", "defer")
            approve_trade: Whether trade is approved
            confidence: Final confidence score
            explanation: Human-readable explanation
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal": signal_data,
            "ml_score": ml_score,
            "llm_decision": llm_decision,
            "final_decision": final_decision,
            "approve_trade": approve_trade,
            "confidence": confidence,
            "explanation": explanation,
        }
        
        if metadata:
            log_entry["metadata"] = metadata
        
        self.decision_logger.info(json.dumps(log_entry))
    
    def log_execution(
        self,
        symbol: str,
        order_id: str,
        client_order_id: str,
        order_type: str,
        side: str,
        price: float,
        size: float,
        status: str,
        filled_size: Optional[float] = None,
        filled_price: Optional[float] = None,
        fee: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a trade execution.
        
        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID
            client_order_id: Client-generated order ID
            order_type: Order type ("limit", "market")
            side: Order side ("buy", "sell")
            price: Order price
            size: Order size
            status: Execution status ("pending", "filled", "partial", "cancelled", "failed")
            filled_size: Filled size (if partial/complete)
            filled_price: Average fill price
            fee: Trading fee paid
            error: Error message if failed
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "order_type": order_type,
            "side": side,
            "price": price,
            "size": size,
            "status": status,
        }
        
        if filled_size is not None:
            log_entry["filled_size"] = filled_size
        if filled_price is not None:
            log_entry["filled_price"] = filled_price
        if fee is not None:
            log_entry["fee"] = fee
        if error:
            log_entry["error"] = error
        if metadata:
            log_entry["metadata"] = metadata
        
        level = logging.ERROR if error else logging.INFO
        self.execution_logger.log(level, json.dumps(log_entry))
    
    def log_performance(
        self,
        total_pnl: float,
        realized_pnl: float,
        unrealized_pnl: float,
        win_rate: float,
        total_trades: int,
        open_positions: int,
        account_balance: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            total_pnl: Total profit/loss
            realized_pnl: Realized profit/loss
            unrealized_pnl: Unrealized profit/loss
            win_rate: Win rate percentage
            total_trades: Total number of trades
            open_positions: Number of open positions
            account_balance: Current account balance
            metadata: Additional metrics
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance",
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "open_positions": open_positions,
            "account_balance": account_balance,
        }
        
        if metadata:
            log_entry["metadata"] = metadata
        
        self.system_logger.info(json.dumps(log_entry))
    
    def info(self, message: str, **kwargs) -> None:
        """Log general info message."""
        self.system_logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.system_logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self.system_logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.system_logger.debug(message, extra=kwargs)


# Global logger instance
_logger_instance: Optional[TrendCortexLogger] = None


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    console_output: bool = True,
    json_format: bool = True,
) -> TrendCortexLogger:
    """
    Initialize global logger instance.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        console_output: Enable console output
        json_format: Use JSON format for logs
        
    Returns:
        Configured logger instance
    """
    global _logger_instance
    
    _logger_instance = TrendCortexLogger(
        log_dir=log_dir,
        level=level,
        console_output=console_output,
        json_format=json_format,
    )
    
    return _logger_instance


def get_logger() -> TrendCortexLogger:
    """
    Get global logger instance.
    
    Returns:
        Global logger instance
        
    Raises:
        RuntimeError: If logger not initialized
    """
    if _logger_instance is None:
        raise RuntimeError(
            "Logger not initialized. Call setup_logging() first."
        )
    
    return _logger_instance
