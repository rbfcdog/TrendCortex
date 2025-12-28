"""
Execution Module

Handles trade order execution including:
- Order placement with validation
- Order tracking and status monitoring
- Order modification and cancellation
- Fill confirmation and error handling
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from trendcortex.api_client import WEEXAPIClient
from trendcortex.config import Config
from trendcortex.logger import get_logger
from trendcortex.risk_controller import RiskAssessment
from trendcortex.signal_engine import TradingSignal
from trendcortex.utils import (
    generate_client_order_id,
    round_to_tick_size,
    round_to_size_increment,
)
from trendcortex.ai_logger import (
    WEEXAILogger,
    create_execution_log,
    serialize_for_ai_log,
)


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderSide(Enum):
    """Order side"""
    BUY = "1"
    SELL = "2"
    CLOSE_LONG = "3"
    CLOSE_SHORT = "4"


@dataclass
class ExecutionResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[str]
    client_order_id: str
    symbol: str
    side: str
    size: float
    price: float
    status: OrderStatus
    filled_size: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = None


class TradeExecutor:
    """
    Handles trade execution with error handling and retry logic.
    
    Supports both live trading and dry-run modes.
    """
    
    def __init__(self, api_client: WEEXAPIClient, config: Config):
        """
        Initialize trade executor.
        
        Args:
            api_client: WEEX API client
            config: System configuration
        """
        self.api = api_client
        self.config = config
        self.logger = get_logger()
        
        # Execution settings
        self.dry_run = config.execution.dry_run
        self.use_limit_orders = config.execution.use_limit_orders
        self.limit_order_offset_percent = config.execution.limit_order_offset_percent
        self.order_timeout = config.execution.order_timeout_seconds
        self.fill_timeout = config.execution.fill_timeout_seconds
        self.slippage_tolerance = config.execution.slippage_tolerance_percent
        
        # Track active orders
        self.active_orders: Dict[str, Dict] = {}
        
        # Dry-run mode tracking
        self.dry_run_orders: List[Dict] = []
        
        # Initialize AI logger for WEEX
        self.ai_logger = WEEXAILogger(
            api_key=config.api.key,
            api_secret=config.api.secret,
            api_passphrase=config.api.passphrase,
            base_url=config.api.base_url,
        )
    
    async def execute_signal(
        self,
        signal: TradingSignal,
        risk_assessment: RiskAssessment,
        contract_specs: Dict,
    ) -> ExecutionResult:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            risk_assessment: Risk assessment with position sizing
            contract_specs: Contract specifications for symbol
            
        Returns:
            Execution result
        """
        # Validate inputs
        if not risk_assessment.approved:
            return ExecutionResult(
                success=False,
                order_id=None,
                client_order_id=generate_client_order_id(),
                symbol=signal.symbol,
                side=signal.direction.value,
                size=0.0,
                price=signal.entry_price,
                status=OrderStatus.FAILED,
                error_message=f"Risk check failed: {risk_assessment.reason}",
                timestamp=datetime.now(),
            )
        
        # Get position size and leverage from risk assessment
        position_size = risk_assessment.suggested_size
        leverage = risk_assessment.suggested_leverage
        
        # Round to valid increments
        position_size = round_to_size_increment(
            position_size,
            contract_specs["size_increment"]
        )
        
        # Validate minimum size
        if position_size < contract_specs["min_order_size"]:
            return ExecutionResult(
                success=False,
                order_id=None,
                client_order_id=generate_client_order_id(),
                symbol=signal.symbol,
                side=signal.direction.value,
                size=position_size,
                price=signal.entry_price,
                status=OrderStatus.FAILED,
                error_message=f"Position size {position_size} below minimum {contract_specs['min_order_size']}",
                timestamp=datetime.now(),
            )
        
        # Set leverage first
        await self._set_leverage(signal.symbol, leverage)
        
        # Determine order side
        order_side = OrderSide.BUY if signal.direction.value == "long" else OrderSide.SELL
        
        # Calculate order price
        if self.use_limit_orders:
            order_price = self._calculate_limit_price(
                signal.entry_price,
                signal.direction.value,
                contract_specs["tick_size"],
            )
        else:
            order_price = None  # Market order
        
        # Execute order
        if self.dry_run:
            result = await self._execute_dry_run(
                signal, order_side, position_size, order_price
            )
        else:
            result = await self._execute_live(
                signal, order_side, position_size, order_price, contract_specs
            )
        
        # Log execution to structured logs
        self.logger.log_execution(
            symbol=signal.symbol,
            order_id=result.order_id or "",
            client_order_id=result.client_order_id,
            order_type="limit" if order_price else "market",
            side=order_side.value,
            price=order_price or signal.entry_price,
            size=position_size,
            status=result.status.value,
            filled_size=result.filled_size,
            filled_price=result.filled_price,
            fee=result.fee,
            error=result.error_message,
            metadata={
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "leverage": leverage,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
            }
        )
        
        # Log execution to WEEX AI
        try:
            log_entry = create_execution_log(
                model="execution-engine-v1",
                input_data=serialize_for_ai_log({
                    "symbol": signal.symbol,
                    "side": order_side.value,
                    "size": position_size,
                    "price": order_price or signal.entry_price,
                    "order_type": "limit" if order_price else "market",
                    "leverage": leverage,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "dry_run": self.dry_run,
                }),
                output_data=serialize_for_ai_log({
                    "order_id": result.order_id,
                    "status": result.status.value,
                    "filled_price": result.filled_price,
                    "filled_size": result.filled_size,
                    "fee": result.fee,
                    "success": result.success,
                }),
                explanation=(
                    f"{'[DRY RUN] ' if self.dry_run else ''}Order executed: "
                    f"{signal.symbol} {order_side.value} {position_size} @ "
                    f"{result.filled_price or order_price or signal.entry_price}. "
                    f"Status: {result.status.value}. "
                    f"{'Success' if result.success else 'Failed: ' + (result.error_message or 'Unknown error')}"
                ),
                order_id=result.order_id,
            )
            await self.ai_logger.upload_log_async(log_entry)
            self.logger.info("Execution logged to WEEX AI")
        except Exception as e:
            self.logger.error(f"Failed to upload execution log: {e}")
        
        return result
    
    async def _execute_dry_run(
        self,
        signal: TradingSignal,
        order_side: OrderSide,
        size: float,
        price: Optional[float],
    ) -> ExecutionResult:
        """
        Simulate order execution in dry-run mode.
        
        Args:
            signal: Trading signal
            order_side: Order side
            size: Order size
            price: Order price (None for market)
            
        Returns:
            Simulated execution result
        """
        client_order_id = generate_client_order_id()
        order_id = f"dry_run_{client_order_id}"
        
        # Simulate order placement
        self.logger.info(
            f"[DRY RUN] Order placed: {signal.symbol} {order_side.value} "
            f"{size} @ {price or signal.entry_price}"
        )
        
        # Store dry-run order
        dry_run_order = {
            "order_id": order_id,
            "client_order_id": client_order_id,
            "symbol": signal.symbol,
            "side": order_side.value,
            "size": size,
            "price": price or signal.entry_price,
            "timestamp": datetime.now(),
        }
        self.dry_run_orders.append(dry_run_order)
        
        # Simulate immediate fill for dry-run
        return ExecutionResult(
            success=True,
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=signal.symbol,
            side=order_side.value,
            size=size,
            price=price or signal.entry_price,
            status=OrderStatus.FILLED,
            filled_size=size,
            filled_price=price or signal.entry_price,
            fee=0.0,  # No fees in dry-run
            timestamp=datetime.now(),
        )
    
    async def _execute_live(
        self,
        signal: TradingSignal,
        order_side: OrderSide,
        size: float,
        price: Optional[float],
        contract_specs: Dict,
    ) -> ExecutionResult:
        """
        Execute order on live exchange.
        
        Args:
            signal: Trading signal
            order_side: Order side
            size: Order size
            price: Order price (None for market)
            contract_specs: Contract specifications
            
        Returns:
            Live execution result
        """
        client_order_id = generate_client_order_id()
        
        try:
            # Round price to valid tick size
            if price:
                price = round_to_tick_size(price, contract_specs["tick_size"])
            
            # Place order
            order_type = "1" if price else "2"  # 1=limit, 2=market
            
            response = await self.api.place_order(
                symbol=signal.symbol,
                side=order_side.value,
                size=str(size),
                order_type=order_type,
                price=str(price) if price else None,
                client_oid=client_order_id,
            )
            
            order_id = response.get("order_id")
            
            self.logger.info(
                f"Order placed: {signal.symbol} {order_side.value} "
                f"{size} @ {price or 'MARKET'} (ID: {order_id})"
            )
            
            # Track order
            self.active_orders[order_id] = {
                "client_order_id": client_order_id,
                "symbol": signal.symbol,
                "side": order_side.value,
                "size": size,
                "price": price,
                "timestamp": datetime.now(),
            }
            
            # Wait for fill confirmation
            fill_result = await self._wait_for_fill(signal.symbol, order_id)
            
            return ExecutionResult(
                success=True,
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=signal.symbol,
                side=order_side.value,
                size=size,
                price=price or signal.entry_price,
                status=fill_result["status"],
                filled_size=fill_result["filled_size"],
                filled_price=fill_result["filled_price"],
                fee=fill_result["fee"],
                timestamp=datetime.now(),
            )
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}", exc_info=True)
            
            return ExecutionResult(
                success=False,
                order_id=None,
                client_order_id=client_order_id,
                symbol=signal.symbol,
                side=order_side.value,
                size=size,
                price=price or signal.entry_price,
                status=OrderStatus.FAILED,
                error_message=str(e),
                timestamp=datetime.now(),
            )
    
    async def _set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Set leverage for symbol.
        
        Args:
            symbol: Trading pair symbol
            leverage: Desired leverage
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Set leverage {symbol}: {leverage}x")
            return
        
        try:
            await self.api.set_leverage(
                symbol=symbol,
                margin_mode=self.config.trading.margin_mode,
                long_leverage=leverage,
                short_leverage=leverage,
            )
            self.logger.info(f"Leverage set for {symbol}: {leverage}x")
        except Exception as e:
            self.logger.warning(f"Failed to set leverage for {symbol}: {e}")
    
    def _calculate_limit_price(
        self,
        entry_price: float,
        direction: str,
        tick_size: float,
    ) -> float:
        """
        Calculate limit order price with small offset for better fill probability.
        
        Args:
            entry_price: Signal entry price
            direction: Order direction
            tick_size: Minimum price increment
            
        Returns:
            Limit order price
        """
        offset = entry_price * (self.limit_order_offset_percent / 100)
        
        if direction == "long":
            # Buy slightly above market for faster fill
            limit_price = entry_price + offset
        else:
            # Sell slightly below market for faster fill
            limit_price = entry_price - offset
        
        # Round to valid tick size
        return round_to_tick_size(limit_price, tick_size)
    
    async def _wait_for_fill(
        self,
        symbol: str,
        order_id: str,
    ) -> Dict:
        """
        Wait for order fill confirmation.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to monitor
            
        Returns:
            Fill information dictionary
        """
        start_time = datetime.now()
        timeout = self.fill_timeout
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                # Check order status
                order_info = await self.api.get_order(symbol, order_id)
                
                status = order_info.get("state")
                
                # Order states: 0=pending, 1=partial, 2=filled, 3=cancelled, 4=failed
                if status in ["2", "filled"]:
                    # Get fill details
                    fills = await self.api.get_trade_fills(symbol, order_id)
                    
                    if fills:
                        total_filled = sum(float(f.get("fillSize", 0)) for f in fills)
                        avg_price = sum(
                            float(f.get("fillSize", 0)) * float(f.get("fillValue", 0))
                            for f in fills
                        ) / total_filled if total_filled > 0 else 0
                        total_fee = sum(float(f.get("fillFee", 0)) for f in fills)
                        
                        return {
                            "status": OrderStatus.FILLED,
                            "filled_size": total_filled,
                            "filled_price": avg_price,
                            "fee": total_fee,
                        }
                
                # Check for cancellation or failure
                if status in ["3", "4", "cancelled", "failed"]:
                    return {
                        "status": OrderStatus.CANCELLED if status in ["3", "cancelled"] else OrderStatus.FAILED,
                        "filled_size": 0.0,
                        "filled_price": 0.0,
                        "fee": 0.0,
                    }
                
            except Exception as e:
                self.logger.warning(f"Error checking order status: {e}")
            
            # Wait before next check
            await asyncio.sleep(2)
        
        # Timeout reached
        self.logger.warning(f"Order fill timeout for {order_id}")
        return {
            "status": OrderStatus.PENDING,
            "filled_size": 0.0,
            "filled_price": 0.0,
            "fee": 0.0,
        }
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Cancel order: {order_id}")
            return True
        
        try:
            await self.api.cancel_order(symbol, order_id)
            
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if successful
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Cancel all orders for {symbol}")
            return True
        
        try:
            await self.api.cancel_all_orders(symbol)
            
            # Clear active orders for symbol
            self.active_orders = {
                k: v for k, v in self.active_orders.items()
                if v["symbol"] != symbol
            }
            
            self.logger.info(f"All orders cancelled for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel orders for {symbol}: {e}")
            return False
    
    def get_dry_run_orders(self) -> List[Dict]:
        """Get list of dry-run orders."""
        return self.dry_run_orders.copy()
