"""
Utility Functions Module

Common helper functions for time conversion, formatting, order sizing,
and other utility operations used throughout the system.
"""

import hashlib
import hmac
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, Tuple

import pytz


def get_current_timestamp_ms() -> int:
    """
    Get current Unix timestamp in milliseconds.
    
    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def get_current_timestamp() -> int:
    """
    Get current Unix timestamp in seconds.
    
    Returns:
        Current timestamp in seconds
    """
    return int(time.time())


def timestamp_to_datetime(timestamp: int, tz: str = "UTC") -> datetime:
    """
    Convert Unix timestamp to datetime object.
    
    Args:
        timestamp: Unix timestamp (seconds or milliseconds)
        tz: Timezone name (default: UTC)
        
    Returns:
        Datetime object in specified timezone
    """
    # Handle both seconds and milliseconds
    if timestamp > 10**10:
        timestamp = timestamp / 1000
    
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    if tz != "UTC":
        target_tz = pytz.timezone(tz)
        dt = dt.astimezone(target_tz)
    
    return dt


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert datetime to Unix timestamp in seconds.
    
    Args:
        dt: Datetime object
        
    Returns:
        Unix timestamp in seconds
    """
    return int(dt.timestamp())


def format_price(price: float, precision: int = 8) -> str:
    """
    Format price with specified precision.
    
    Args:
        price: Price value
        precision: Number of decimal places
        
    Returns:
        Formatted price string
    """
    return f"{price:.{precision}f}".rstrip("0").rstrip(".")


def format_size(size: float, precision: int = 4) -> str:
    """
    Format order size with specified precision.
    
    Args:
        size: Order size
        precision: Number of decimal places
        
    Returns:
        Formatted size string
    """
    return f"{size:.{precision}f}".rstrip("0").rstrip(".")


def round_down(value: float, decimals: int) -> float:
    """
    Round down a value to specified decimal places.
    
    Args:
        value: Value to round
        decimals: Number of decimal places
        
    Returns:
        Rounded down value
    """
    multiplier = 10 ** decimals
    return float(int(value * multiplier) / multiplier)


def round_to_tick_size(price: float, tick_size: float) -> float:
    """
    Round price to valid tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum price increment
        
    Returns:
        Price rounded to valid tick size
    """
    return round(price / tick_size) * tick_size


def round_to_size_increment(size: float, size_increment: float) -> float:
    """
    Round size to valid size increment.
    
    Args:
        size: Size to round
        size_increment: Minimum size increment
        
    Returns:
        Size rounded to valid increment
    """
    decimal_places = len(str(size_increment).split('.')[-1]) if '.' in str(size_increment) else 0
    return round_down(size / size_increment, 0) * size_increment


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: int = 1,
    max_size: Optional[float] = None,
) -> float:
    """
    Calculate position size based on risk management rules.
    
    Args:
        account_balance: Available account balance in USDT
        risk_percent: Percentage of account to risk (0-100)
        entry_price: Entry price for position
        stop_loss_price: Stop loss price
        leverage: Trading leverage
        max_size: Maximum allowed position size
        
    Returns:
        Calculated position size
    """
    # Calculate risk amount in USDT
    risk_amount = account_balance * (risk_percent / 100)
    
    # Calculate price difference percentage
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0.0
    
    price_diff_percent = abs(entry_price - stop_loss_price) / entry_price * 100
    
    if price_diff_percent == 0:
        return 0.0
    
    # Calculate position size
    position_value = risk_amount / (price_diff_percent / 100)
    position_size = (position_value / entry_price) * leverage
    
    # Apply max size limit if specified
    if max_size and position_size > max_size:
        position_size = max_size
    
    return position_size


def calculate_pnl(
    entry_price: float,
    current_price: float,
    size: float,
    side: str,
    leverage: int = 1,
) -> float:
    """
    Calculate profit/loss for a position.
    
    Args:
        entry_price: Entry price
        current_price: Current market price
        size: Position size
        side: Position side ("long" or "short")
        leverage: Trading leverage
        
    Returns:
        PnL in USDT
    """
    if side.lower() == "long":
        pnl_percent = (current_price - entry_price) / entry_price
    else:  # short
        pnl_percent = (entry_price - current_price) / entry_price
    
    position_value = size * entry_price
    pnl = position_value * pnl_percent * leverage
    
    return pnl


def calculate_liquidation_price(
    entry_price: float,
    leverage: int,
    side: str,
    maintenance_margin_rate: float = 0.005,
) -> float:
    """
    Calculate liquidation price for a leveraged position.
    
    Args:
        entry_price: Entry price
        leverage: Trading leverage
        side: Position side ("long" or "short")
        maintenance_margin_rate: Maintenance margin rate (default 0.5%)
        
    Returns:
        Liquidation price
    """
    if side.lower() == "long":
        liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
    else:  # short
        liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin_rate)
    
    return liquidation_price


def calculate_leverage_from_margin(
    position_value: float,
    margin: float,
) -> float:
    """
    Calculate effective leverage from position value and margin.
    
    Args:
        position_value: Total position value in USDT
        margin: Margin used in USDT
        
    Returns:
        Effective leverage
    """
    if margin == 0:
        return 0.0
    
    return position_value / margin


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage value (e.g., 5.5 for 5.5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def calculate_fee(
    size: float,
    price: float,
    fee_rate: float,
) -> float:
    """
    Calculate trading fee.
    
    Args:
        size: Trade size
        price: Trade price
        fee_rate: Fee rate (e.g., 0.0008 for 0.08%)
        
    Returns:
        Fee amount in USDT
    """
    trade_value = size * price
    return trade_value * fee_rate


def generate_client_order_id(prefix: str = "tc") -> str:
    """
    Generate unique client order ID.
    
    Args:
        prefix: Prefix for order ID
        
    Returns:
        Unique order ID string
    """
    timestamp = get_current_timestamp_ms()
    return f"{prefix}_{timestamp}"


def validate_symbol(symbol: str, allowed_symbols: list) -> bool:
    """
    Validate if symbol is allowed for trading.
    
    Args:
        symbol: Trading pair symbol
        allowed_symbols: List of allowed symbols
        
    Returns:
        True if symbol is allowed
    """
    return symbol.lower() in [s.lower() for s in allowed_symbols]


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
        
    Returns:
        Number of seconds
        
    Raises:
        ValueError: If timeframe format is invalid
    """
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }
    
    if unit not in multipliers:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    
    return value * multipliers[unit]


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def normalize_side(side: str) -> str:
    """
    Normalize order side to standard format.
    
    Args:
        side: Order side (various formats)
        
    Returns:
        Normalized side ("buy" or "sell")
    """
    side_lower = side.lower()
    
    if side_lower in ("buy", "long", "1"):
        return "buy"
    elif side_lower in ("sell", "short", "2"):
        return "sell"
    else:
        raise ValueError(f"Invalid side: {side}")


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    side: str,
) -> float:
    """
    Calculate risk/reward ratio for a trade.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        side: Trade side ("long" or "short")
        
    Returns:
        Risk/reward ratio
    """
    if side.lower() == "long":
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # short
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    
    if risk <= 0:
        return 0.0
    
    return reward / risk


def exponential_backoff_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay for retry attempts.
    
    Args:
        attempt: Retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def dict_to_query_string(params: Dict) -> str:
    """
    Convert dictionary to URL query string.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Query string (without leading '?')
    """
    if not params:
        return ""
    
    return "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
