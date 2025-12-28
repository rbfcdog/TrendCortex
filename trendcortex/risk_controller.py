"""
Risk Controller Module

Enforces trading risk management rules including:
- Position sizing
- Leverage limits
- Stop loss/take profit
- Maximum drawdown
- Cooldown periods
- Volatility filters
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from trendcortex.config import Config
from trendcortex.logger import get_logger
from trendcortex.signal_engine import TradingSignal
from trendcortex.utils import calculate_position_size, calculate_pnl


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    approved: bool
    reason: str
    suggested_size: Optional[float] = None
    suggested_leverage: Optional[int] = None
    warnings: List[str] = None


class RiskController:
    """
    Multi-layer risk management system.
    
    Validates all trades against comprehensive risk rules before execution.
    """
    
    def __init__(self, config: Config):
        """
        Initialize risk controller.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = get_logger()
        
        # Risk limits
        self.max_position_size_usdt = config.risk.max_position_size_usdt
        self.max_position_size_percent = config.risk.max_position_size_percent
        self.max_leverage = config.risk.max_leverage
        self.min_leverage = config.risk.min_leverage
        self.max_open_positions = config.risk.max_open_positions
        self.max_positions_per_symbol = config.risk.max_positions_per_symbol
        self.stop_loss_percent = config.risk.stop_loss_percent
        self.take_profit_percent = config.risk.take_profit_percent
        self.max_portfolio_risk_percent = config.risk.max_portfolio_risk_percent
        self.max_daily_loss_percent = config.risk.max_daily_loss_percent
        self.cooldown_seconds = config.risk.cooldown_seconds
        self.min_account_balance = config.risk.min_account_balance_usdt
        
        # Volatility filter
        self.volatility_filter_enabled = config.risk.volatility_filter["enabled"]
        self.max_atr_percent = config.risk.volatility_filter["max_atr_percent"]
        
        # Track recent trades for cooldown
        self.recent_trades: Dict[str, List[float]] = {}  # symbol -> [timestamps]
        
        # Track daily PnL
        self.daily_pnl_start_balance: Optional[float] = None
        self.daily_pnl_reset_date: Optional[datetime] = None
    
    async def validate_trade(
        self,
        signal: TradingSignal,
        account_balance: float,
        open_positions: List[Dict],
        current_volatility: float,
    ) -> RiskAssessment:
        """
        Validate if trade passes all risk checks.
        
        Args:
            signal: Trading signal
            account_balance: Available account balance
            open_positions: List of currently open positions
            current_volatility: Current market volatility (ATR %)
            
        Returns:
            Risk assessment with approval decision
        """
        warnings = []
        
        # Check 1: Minimum account balance
        if account_balance < self.min_account_balance:
            return RiskAssessment(
                approved=False,
                reason=f"Account balance ${account_balance:.2f} below minimum ${self.min_account_balance:.2f}",
            )
        
        # Check 2: Maximum open positions
        if len(open_positions) >= self.max_open_positions:
            return RiskAssessment(
                approved=False,
                reason=f"Maximum open positions reached ({len(open_positions)}/{self.max_open_positions})",
            )
        
        # Check 3: Maximum positions per symbol
        symbol_positions = [p for p in open_positions if p.get("symbol") == signal.symbol]
        if len(symbol_positions) >= self.max_positions_per_symbol:
            return RiskAssessment(
                approved=False,
                reason=f"Maximum positions for {signal.symbol} reached ({len(symbol_positions)}/{self.max_positions_per_symbol})",
            )
        
        # Check 4: Cooldown period
        if not self._check_cooldown(signal.symbol):
            time_remaining = self._get_cooldown_remaining(signal.symbol)
            return RiskAssessment(
                approved=False,
                reason=f"Cooldown active for {signal.symbol}: {time_remaining:.0f}s remaining",
            )
        
        # Check 5: Volatility filter
        if self.volatility_filter_enabled and current_volatility > self.max_atr_percent:
            return RiskAssessment(
                approved=False,
                reason=f"Volatility too high: {current_volatility:.2f}% > {self.max_atr_percent:.2f}%",
            )
        
        # Check 6: Daily loss limit
        if not self._check_daily_loss_limit(account_balance):
            return RiskAssessment(
                approved=False,
                reason=f"Daily loss limit exceeded",
            )
        
        # Check 7: Validate stop loss and take profit
        if signal.stop_loss is None or signal.take_profit is None:
            warnings.append("No stop loss or take profit set")
        else:
            # Ensure stop loss and take profit are reasonable
            sl_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
            tp_distance = abs(signal.take_profit - signal.entry_price) / signal.entry_price * 100
            
            if sl_distance < 0.1 or sl_distance > 10:
                warnings.append(f"Stop loss distance unusual: {sl_distance:.2f}%")
            
            if tp_distance < 0.2 or tp_distance > 20:
                warnings.append(f"Take profit distance unusual: {tp_distance:.2f}%")
        
        # Calculate suggested position size
        suggested_size, suggested_leverage = self._calculate_safe_position_size(
            signal=signal,
            account_balance=account_balance,
            open_positions=open_positions,
        )
        
        if suggested_size <= 0:
            return RiskAssessment(
                approved=False,
                reason="Calculated position size is zero or negative",
            )
        
        # Check leverage limits
        if suggested_leverage > self.max_leverage:
            warnings.append(f"Leverage capped at {self.max_leverage}x (requested {suggested_leverage}x)")
            suggested_leverage = self.max_leverage
        
        if suggested_leverage < self.min_leverage:
            suggested_leverage = self.min_leverage
        
        return RiskAssessment(
            approved=True,
            reason="All risk checks passed",
            suggested_size=suggested_size,
            suggested_leverage=suggested_leverage,
            warnings=warnings if warnings else None,
        )
    
    def _check_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if cooldown has elapsed
        """
        if symbol not in self.recent_trades:
            return True
        
        current_time = time.time()
        
        # Remove expired entries
        self.recent_trades[symbol] = [
            t for t in self.recent_trades[symbol]
            if current_time - t < self.cooldown_seconds
        ]
        
        return len(self.recent_trades[symbol]) == 0
    
    def _get_cooldown_remaining(self, symbol: str) -> float:
        """Get remaining cooldown time in seconds."""
        if symbol not in self.recent_trades or not self.recent_trades[symbol]:
            return 0.0
        
        last_trade_time = max(self.recent_trades[symbol])
        elapsed = time.time() - last_trade_time
        remaining = self.cooldown_seconds - elapsed
        
        return max(remaining, 0.0)
    
    def record_trade(self, symbol: str) -> None:
        """
        Record a trade execution for cooldown tracking.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = []
        
        self.recent_trades[symbol].append(time.time())
        self.logger.debug(f"Trade recorded for cooldown: {symbol}")
    
    def _check_daily_loss_limit(self, current_balance: float) -> bool:
        """
        Check if daily loss limit has been exceeded.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            True if within daily loss limit
        """
        # Reset daily tracking at start of new day
        today = datetime.now().date()
        if self.daily_pnl_reset_date != today:
            self.daily_pnl_start_balance = current_balance
            self.daily_pnl_reset_date = today
            return True
        
        # Calculate daily PnL
        if self.daily_pnl_start_balance is None:
            self.daily_pnl_start_balance = current_balance
            return True
        
        daily_pnl = current_balance - self.daily_pnl_start_balance
        daily_pnl_percent = (daily_pnl / self.daily_pnl_start_balance) * 100
        
        # Check if loss exceeds limit
        if daily_pnl_percent < -self.max_daily_loss_percent:
            self.logger.warning(
                f"Daily loss limit reached: {daily_pnl_percent:.2f}% < -{self.max_daily_loss_percent:.2f}%"
            )
            return False
        
        return True
    
    def _calculate_safe_position_size(
        self,
        signal: TradingSignal,
        account_balance: float,
        open_positions: List[Dict],
    ) -> Tuple[float, int]:
        """
        Calculate safe position size based on risk parameters.
        
        Args:
            signal: Trading signal
            account_balance: Available balance
            open_positions: Current open positions
            
        Returns:
            Tuple of (position_size, leverage)
        """
        # Calculate risk per trade
        risk_amount = account_balance * (self.max_portfolio_risk_percent / 100)
        
        # Adjust for existing positions (reduce risk if portfolio is exposed)
        if open_positions:
            exposure_factor = 1.0 - (len(open_positions) / self.max_open_positions) * 0.5
            risk_amount *= exposure_factor
        
        # Calculate position size based on stop loss
        if signal.stop_loss:
            stop_loss_distance = abs(signal.entry_price - signal.stop_loss)
            stop_loss_percent = (stop_loss_distance / signal.entry_price) * 100
            
            # Position value that risks the calculated risk amount
            position_value = risk_amount / (stop_loss_percent / 100)
            position_size = position_value / signal.entry_price
        else:
            # Fallback: use default risk percentage
            position_value = account_balance * (self.max_position_size_percent / 100)
            position_size = position_value / signal.entry_price
        
        # Determine leverage
        # Start with default, adjust based on confidence
        leverage = self.config.trading.default_leverage
        
        # Reduce leverage for lower confidence signals
        if signal.confidence < 0.7:
            leverage = max(self.min_leverage, leverage // 2)
        
        # Cap at maximum
        leverage = min(leverage, self.max_leverage)
        
        # Apply position size limits
        max_size_from_usdt = self.max_position_size_usdt / signal.entry_price
        position_size = min(position_size, max_size_from_usdt)
        
        return position_size, leverage
    
    def calculate_portfolio_risk(
        self,
        open_positions: List[Dict],
        current_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate current portfolio risk metrics.
        
        Args:
            open_positions: List of open positions
            current_prices: Current market prices
            
        Returns:
            Risk metrics dictionary
        """
        total_value = 0.0
        total_unrealized_pnl = 0.0
        position_risks = []
        
        for position in open_positions:
            symbol = position["symbol"]
            entry_price = position["entry_price"]
            size = position["size"]
            side = position["side"]
            leverage = position.get("leverage", 1)
            
            current_price = current_prices.get(symbol, entry_price)
            
            # Calculate position value and PnL
            position_value = size * entry_price
            unrealized_pnl = calculate_pnl(
                entry_price, current_price, size, side, leverage
            )
            
            total_value += position_value
            total_unrealized_pnl += unrealized_pnl
            
            # Calculate risk (distance to stop loss)
            stop_loss = position.get("stop_loss")
            if stop_loss:
                max_loss = abs(entry_price - stop_loss) * size * leverage
                position_risks.append(max_loss)
        
        total_risk = sum(position_risks)
        
        return {
            "total_position_value": total_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_risk_exposure": total_risk,
            "num_positions": len(open_positions),
        }
