"""
AI Logger Module for WEEX OpenAPI Integration

This module provides functionality to log AI model decisions and actions
to the WEEX AI log endpoint for competition tracking and transparency.

Author: TrendCortex Team
Competition: WEEX AI Wars - Alpha Awakens
"""

import json
import hmac
import hashlib
import time
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import aiohttp
import asyncio
from decimal import Decimal

from .logger import get_logger


# Module logger
logger = get_logger(__name__)


class AILogStage(Enum):
    """AI workflow stages for WEEX logging"""
    STRATEGY_GENERATION = "Strategy Generation"
    DECISION_MAKING = "Decision Making"
    EXECUTION = "Execution"
    RISK_ASSESSMENT = "Risk Assessment"
    SIGNAL_ANALYSIS = "Signal Analysis"


@dataclass
class AILogEntry:
    """
    Structured AI log entry matching WEEX requirements.
    
    Attributes:
        stage: The workflow stage (Strategy Generation, Decision Making, Execution)
        model: Model name or version used (e.g., "gpt-4", "xgboost-v1.2", "hybrid-v1")
        input_data: Input data fed to the model (dict or string)
        output_data: Model output/decision (dict or string)
        explanation: Human-readable explanation of the decision
        order_id: Optional order ID if associated with a specific order
        timestamp: Timestamp of the log entry (auto-generated)
    """
    stage: str
    model: str
    input_data: Union[Dict[str, Any], str]
    output_data: Union[Dict[str, Any], str]
    explanation: str
    order_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_weex_payload(self) -> Dict[str, Any]:
        """
        Convert log entry to WEEX API payload format.
        
        Returns:
            Dictionary matching WEEX uploadAiLog requirements
        """
        payload = {
            "stage": self.stage,
            "model": self.model,
            "input": self._serialize_data(self.input_data),
            "output": self._serialize_data(self.output_data),
            "explanation": self.explanation,
        }
        
        # Add optional order_id if present
        if self.order_id:
            payload["orderId"] = self.order_id
        
        return payload
    
    @staticmethod
    def _serialize_data(data: Union[Dict, str, Any]) -> str:
        """
        Safely serialize data to JSON string.
        
        Args:
            data: Data to serialize (dict, string, or other)
            
        Returns:
            JSON string representation
        """
        if isinstance(data, str):
            return data
        
        try:
            # Handle special types like Decimal, datetime, etc.
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize data, using string representation: {e}")
            return str(data)


class WEEXAILogger:
    """
    WEEX AI Logger client for uploading AI decision logs.
    
    This class handles authentication, signing, and uploading of AI logs
    to the WEEX OpenAPI endpoint: POST /capi/v2/order/uploadAiLog
    """
    
    UPLOAD_ENDPOINT = "/capi/v2/order/uploadAiLog"
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        base_url: str = "https://api-contract.weex.com",
        timeout: int = 10,
    ):
        """
        Initialize WEEX AI Logger.
        
        Args:
            api_key: WEEX API key (ACCESS-KEY)
            api_secret: WEEX API secret for signing
            api_passphrase: WEEX API passphrase
            base_url: WEEX API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        logger.info("WEEX AI Logger initialized")
    
    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = ""
    ) -> str:
        """
        Generate HMAC SHA256 signature for WEEX API request.
        
        The signature is created by:
        1. Creating a prehash string: timestamp + method + request_path + body
        2. HMAC SHA256 signing with API secret
        3. Base64 encoding the result
        
        Args:
            timestamp: Unix timestamp in milliseconds (as string)
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body (JSON string for POST)
            
        Returns:
            Base64-encoded HMAC SHA256 signature
        """
        # Construct the prehash string
        prehash = f"{timestamp}{method}{request_path}{body}"
        
        # Create HMAC SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            prehash.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        # Base64 encode
        import base64
        signature_b64 = base64.b64encode(signature).decode('utf-8')
        
        return signature_b64
    
    def _get_headers(self, timestamp: str, signature: str) -> Dict[str, str]:
        """
        Get HTTP headers for WEEX API request.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            signature: Generated signature
            
        Returns:
            Dictionary of headers
        """
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.api_passphrase,
            "Content-Type": "application/json",
        }
    
    async def upload_log_async(self, log_entry: AILogEntry) -> Dict[str, Any]:
        """
        Upload AI log entry to WEEX asynchronously.
        
        Args:
            log_entry: AILogEntry object to upload
            
        Returns:
            Response from WEEX API
            
        Raises:
            aiohttp.ClientError: If request fails
            ValueError: If response indicates error
        """
        # Generate timestamp (milliseconds)
        timestamp = str(int(time.time() * 1000))
        
        # Prepare request body
        payload = log_entry.to_weex_payload()
        body = json.dumps(payload, ensure_ascii=False)
        
        # Generate signature
        signature = self._generate_signature(
            timestamp=timestamp,
            method="POST",
            request_path=self.UPLOAD_ENDPOINT,
            body=body
        )
        
        # Prepare headers
        headers = self._get_headers(timestamp, signature)
        
        # Full URL
        url = f"{self.base_url}{self.UPLOAD_ENDPOINT}"
        
        logger.info(f"Uploading AI log: stage={log_entry.stage}, model={log_entry.model}")
        logger.debug(f"AI log payload: {payload}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_text = await response.text()
                    
                    # Log response
                    logger.debug(f"WEEX AI log response: {response.status} - {response_text}")
                    
                    # Parse response
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = {"raw_response": response_text}
                    
                    # Check for errors
                    if response.status != 200:
                        error_msg = f"Failed to upload AI log: {response.status} - {response_text}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    # WEEX API returns {"code": "0", "msg": "success", ...} on success
                    if isinstance(response_data, dict):
                        code = response_data.get("code", "")
                        msg = response_data.get("msg", "")
                        
                        if code != "0" and code != 0:
                            error_msg = f"WEEX API error: code={code}, msg={msg}"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    
                    logger.info(f"AI log uploaded successfully: {response_data}")
                    return response_data
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error uploading AI log: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading AI log: {e}")
            raise
    
    def upload_log_sync(self, log_entry: AILogEntry) -> Dict[str, Any]:
        """
        Upload AI log entry to WEEX synchronously.
        
        This is a wrapper around the async method for use in synchronous code.
        
        Args:
            log_entry: AILogEntry object to upload
            
        Returns:
            Response from WEEX API
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.upload_log_async(log_entry)
                )
                return future.result()
        else:
            # Run in current event loop
            return loop.run_until_complete(self.upload_log_async(log_entry))


# ============================================================================
# Helper Functions
# ============================================================================

def create_strategy_log(
    model: str,
    input_data: Union[Dict, str],
    output_data: Union[Dict, str],
    explanation: str,
    order_id: Optional[str] = None
) -> AILogEntry:
    """
    Create an AI log entry for Strategy Generation stage.
    
    Args:
        model: Model name (e.g., "trend-analyzer-v1")
        input_data: Market data, indicators, etc.
        output_data: Generated strategy/signals
        explanation: Why this strategy was chosen
        order_id: Optional order ID
        
    Returns:
        AILogEntry ready to upload
        
    Example:
        >>> log = create_strategy_log(
        ...     model="ema-crossover-v1",
        ...     input_data={"ema_fast": 12, "ema_slow": 26, "price": 45000},
        ...     output_data={"signal": "BUY", "confidence": 0.85},
        ...     explanation="Fast EMA crossed above slow EMA with strong volume"
        ... )
    """
    return AILogEntry(
        stage=AILogStage.STRATEGY_GENERATION.value,
        model=model,
        input_data=input_data,
        output_data=output_data,
        explanation=explanation,
        order_id=order_id
    )


def create_decision_log(
    model: str,
    input_data: Union[Dict, str],
    output_data: Union[Dict, str],
    explanation: str,
    order_id: Optional[str] = None
) -> AILogEntry:
    """
    Create an AI log entry for Decision Making stage.
    
    Args:
        model: Model name (e.g., "gpt-4", "claude-3", "xgboost-v2")
        input_data: Signal, market context, risk parameters
        output_data: Decision (go/no-go, position size, etc.)
        explanation: Reasoning behind the decision
        order_id: Optional order ID
        
    Returns:
        AILogEntry ready to upload
        
    Example:
        >>> log = create_decision_log(
        ...     model="gpt-4-turbo",
        ...     input_data={
        ...         "signal": "BUY",
        ...         "confidence": 0.85,
        ...         "rsi": 35,
        ...         "market_context": "oversold bounce"
        ...     },
        ...     output_data={
        ...         "decision": "APPROVE",
        ...         "position_size": 0.1,
        ...         "reason": "high confidence with confirmed oversold"
        ...     },
        ...     explanation="LLM approved trade based on strong technical setup"
        ... )
    """
    return AILogEntry(
        stage=AILogStage.DECISION_MAKING.value,
        model=model,
        input_data=input_data,
        output_data=output_data,
        explanation=explanation,
        order_id=order_id
    )


def create_execution_log(
    model: str,
    input_data: Union[Dict, str],
    output_data: Union[Dict, str],
    explanation: str,
    order_id: Optional[str] = None
) -> AILogEntry:
    """
    Create an AI log entry for Execution stage.
    
    Args:
        model: Model name (e.g., "execution-engine-v1")
        input_data: Order parameters, market conditions
        output_data: Execution result, order details
        explanation: How execution was performed
        order_id: Optional order ID
        
    Returns:
        AILogEntry ready to upload
        
    Example:
        >>> log = create_execution_log(
        ...     model="smart-router-v1",
        ...     input_data={
        ...         "symbol": "BTCUSDT",
        ...         "side": "BUY",
        ...         "size": 0.1,
        ...         "order_type": "LIMIT"
        ...     },
        ...     output_data={
        ...         "order_id": "123456",
        ...         "filled_price": 44950,
        ...         "filled_size": 0.1,
        ...         "status": "FILLED"
        ...     },
        ...     explanation="Limit order placed and filled at favorable price",
        ...     order_id="123456"
        ... )
    """
    return AILogEntry(
        stage=AILogStage.EXECUTION.value,
        model=model,
        input_data=input_data,
        output_data=output_data,
        explanation=explanation,
        order_id=order_id
    )


def create_risk_assessment_log(
    model: str,
    input_data: Union[Dict, str],
    output_data: Union[Dict, str],
    explanation: str,
    order_id: Optional[str] = None
) -> AILogEntry:
    """
    Create an AI log entry for Risk Assessment stage.
    
    Args:
        model: Model name (e.g., "risk-analyzer-v1")
        input_data: Position details, account state, market volatility
        output_data: Risk assessment result
        explanation: Risk evaluation reasoning
        order_id: Optional order ID
        
    Returns:
        AILogEntry ready to upload
    """
    return AILogEntry(
        stage=AILogStage.RISK_ASSESSMENT.value,
        model=model,
        input_data=input_data,
        output_data=output_data,
        explanation=explanation,
        order_id=order_id
    )


def serialize_for_ai_log(data: Any) -> Union[Dict, str]:
    """
    Safely serialize any data type for AI log input/output.
    
    Handles:
    - Dictionaries (pass through)
    - Strings (pass through)
    - Pandas DataFrames (convert to dict)
    - Numpy arrays (convert to list)
    - Decimal (convert to float)
    - Datetime (convert to ISO string)
    - Other objects (convert to string)
    
    Args:
        data: Data to serialize
        
    Returns:
        Serialized data (dict or string)
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"price": [100, 101, 102]})
        >>> serialized = serialize_for_ai_log(df)
        >>> isinstance(serialized, dict)
        True
    """
    # Already a dict or string
    if isinstance(data, (dict, str)):
        return data
    
    # Pandas DataFrame
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
        if isinstance(data, pd.Series):
            return data.to_dict()
    except ImportError:
        pass
    
    # Numpy array
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return data.tolist()
    except ImportError:
        pass
    
    # Decimal
    if isinstance(data, Decimal):
        return float(data)
    
    # Datetime
    if isinstance(data, datetime):
        return data.isoformat()
    
    # List or tuple
    if isinstance(data, (list, tuple)):
        return [serialize_for_ai_log(item) for item in data]
    
    # Fallback to string
    return str(data)


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """
    Example usage of the WEEX AI Logger.
    
    This demonstrates how to:
    1. Initialize the logger
    2. Log a decision making event (after LLM call)
    3. Log an execution event (after placing order)
    """
    
    # TODO: Replace with your actual WEEX API credentials
    api_key = "your_api_key_here"
    api_secret = "your_api_secret_here"
    api_passphrase = "your_api_passphrase_here"
    
    # Initialize AI logger
    ai_logger = WEEXAILogger(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase
    )
    
    # ========================================================================
    # Example 1: Log Decision Making Stage (after LLM call)
    # ========================================================================
    
    # Simulate LLM decision making
    llm_input = {
        "signal": {
            "type": "BUY",
            "symbol": "BTCUSDT",
            "confidence": 0.87,
            "price": 44950,
        },
        "market_context": {
            "rsi": 32,
            "trend": "bullish",
            "support_level": 44500,
            "resistance_level": 46000,
        },
        "risk_metrics": {
            "account_balance": 10000,
            "max_position_size": 0.5,
            "stop_loss_pct": 2.0,
        }
    }
    
    llm_output = {
        "decision": "APPROVE",
        "position_size": 0.3,
        "leverage": 5,
        "reasoning": "Strong oversold signal with bullish divergence",
        "confidence_adjustment": 0.92,
        "risk_reward_ratio": 3.5,
    }
    
    # Create decision log
    decision_log = create_decision_log(
        model="gpt-4-turbo-2024-12",
        input_data=serialize_for_ai_log(llm_input),
        output_data=serialize_for_ai_log(llm_output),
        explanation=(
            "LLM approved BUY signal with 92% confidence. "
            "Technical indicators show oversold conditions with bullish divergence. "
            "Risk-reward ratio of 3.5:1 justifies position size of 0.3 BTC with 5x leverage."
        )
    )
    
    # Upload decision log
    try:
        response = await ai_logger.upload_log_async(decision_log)
        print(f"Decision log uploaded: {response}")
    except Exception as e:
        print(f"Failed to upload decision log: {e}")
    
    # ========================================================================
    # Example 2: Log Execution Stage (after placing order)
    # ========================================================================
    
    # Simulate order placement
    order_params = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "LIMIT",
        "price": 44950,
        "size": 0.3,
        "leverage": 5,
        "time_in_force": "GTC",
    }
    
    execution_result = {
        "order_id": "987654321",
        "status": "FILLED",
        "filled_price": 44945,  # Got better price!
        "filled_size": 0.3,
        "fill_time": datetime.utcnow().isoformat(),
        "commission": 0.0002,
        "slippage": -5,  # Negative = favorable
    }
    
    # Create execution log
    execution_log = create_execution_log(
        model="smart-execution-v1",
        input_data=serialize_for_ai_log(order_params),
        output_data=serialize_for_ai_log(execution_result),
        explanation=(
            "Limit order placed at 44950 and filled at 44945, "
            "saving $1.50 in slippage. Order fully filled within 2 seconds. "
            "Position now open: 0.3 BTC LONG with 5x leverage."
        ),
        order_id="987654321"  # Link to the actual order
    )
    
    # Upload execution log
    try:
        response = await ai_logger.upload_log_async(execution_log)
        print(f"Execution log uploaded: {response}")
    except Exception as e:
        print(f"Failed to upload execution log: {e}")
    
    # ========================================================================
    # Example 3: Log Strategy Generation Stage
    # ========================================================================
    
    strategy_input = {
        "timeframe": "15m",
        "indicators": {
            "ema_12": 44800,
            "ema_26": 44600,
            "rsi": 32,
            "macd": 50,
            "volume_ratio": 1.8,
        },
        "price": 44950,
    }
    
    strategy_output = {
        "signal": "BUY",
        "confidence": 0.87,
        "entry_price": 44950,
        "stop_loss": 44050,
        "take_profit_1": 46000,
        "take_profit_2": 47500,
        "strategy_name": "EMA Crossover + RSI Oversold",
    }
    
    strategy_log = create_strategy_log(
        model="technical-analyzer-v2.1",
        input_data=serialize_for_ai_log(strategy_input),
        output_data=serialize_for_ai_log(strategy_output),
        explanation=(
            "EMA 12 crossed above EMA 26 with RSI at oversold level (32). "
            "Volume is 1.8x average, confirming strong buying pressure. "
            "Multi-target strategy with 2% stop loss and staged exits."
        )
    )
    
    try:
        response = await ai_logger.upload_log_async(strategy_log)
        print(f"Strategy log uploaded: {response}")
    except Exception as e:
        print(f"Failed to upload strategy log: {e}")


# Synchronous example
def example_usage_sync():
    """
    Example usage with synchronous code.
    """
    # TODO: Replace with your actual WEEX API credentials
    ai_logger = WEEXAILogger(
        api_key="your_api_key",
        api_secret="your_api_secret",
        api_passphrase="your_api_passphrase"
    )
    
    # Create a simple decision log
    log = create_decision_log(
        model="simple-model-v1",
        input_data={"signal": "BUY", "price": 45000},
        output_data={"decision": "APPROVE", "size": 0.1},
        explanation="Test log entry"
    )
    
    # Upload synchronously
    try:
        response = ai_logger.upload_log_sync(log)
        print(f"Log uploaded: {response}")
    except Exception as e:
        print(f"Failed to upload log: {e}")


if __name__ == "__main__":
    """
    Run examples if this module is executed directly.
    """
    print("=" * 80)
    print("WEEX AI Logger - Example Usage")
    print("=" * 80)
    print("\nRunning async example...")
    
    # Run async example
    asyncio.run(example_usage())
    
    print("\n" + "=" * 80)
    print("Examples complete! Remember to:")
    print("1. Replace API credentials with your actual WEEX keys")
    print("2. Integrate this module into your trading bot")
    print("3. Call upload_log_async() or upload_log_sync() at appropriate stages")
    print("=" * 80)
