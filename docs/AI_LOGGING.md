# WEEX AI Logging Integration

## Overview

TrendCortex integrates with the WEEX AI Wars competition logging system to provide transparency and auditability of all AI-driven trading decisions. This document explains how the AI logging system works and how to use it.

## Table of Contents

1. [What is AI Logging?](#what-is-ai-logging)
2. [Architecture](#architecture)
3. [Usage Guide](#usage-guide)
4. [API Reference](#api-reference)
5. [Integration Examples](#integration-examples)
6. [Troubleshooting](#troubleshooting)

---

## What is AI Logging?

The WEEX AI Wars competition requires all AI trading bots to log their decision-making processes. This creates transparency and allows judges to understand:

- **Strategy Generation**: How signals are generated from market data
- **Decision Making**: How AI models (ML/LLM) evaluate and approve trades
- **Execution**: How orders are placed and filled

Each log entry includes:
- **Model**: Which AI model made the decision
- **Input**: What data was fed to the model
- **Output**: What decision the model made
- **Explanation**: Human-readable reasoning
- **Order ID**: Link to actual orders (optional)

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     TrendCortex Bot                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   Signal     │────>│   ML Model   │────>│  Execution  │ │
│  │   Engine     │     │ Integration  │     │   Engine    │ │
│  └──────┬───────┘     └──────┬───────┘     └──────┬──────┘ │
│         │                    │                     │         │
│         v                    v                     v         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              WEEX AI Logger Module                   │  │
│  │  (ai_logger.py)                                      │  │
│  └────────────────────────┬─────────────────────────────┘  │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            v
              ┌─────────────────────────┐
              │  WEEX API Endpoint      │
              │  POST /capi/v2/order/   │
              │       uploadAiLog       │
              └─────────────────────────┘
```

### Log Stages

1. **Strategy Generation** 
   - When: After technical indicators generate a signal
   - Where: `signal_engine.py` → `ai_logger.py`
   - Purpose: Show how market data led to a trading signal

2. **Decision Making**
   - When: After ML/LLM evaluates the signal
   - Where: `model_integration.py` → `ai_logger.py`
   - Purpose: Show AI reasoning for approving/rejecting trade

3. **Execution**
   - When: After order is placed and filled
   - Where: `execution.py` → `ai_logger.py`
   - Purpose: Show actual execution results

---

## Usage Guide

### Basic Setup

The AI logger is automatically initialized in your bot with your WEEX API credentials:

```python
from trendcortex.ai_logger import WEEXAILogger

# Initialize logger (done automatically in bot)
ai_logger = WEEXAILogger(
    api_key="your_key",
    api_secret="your_secret",
    api_passphrase="your_passphrase"
)
```

### Logging a Strategy Generation

When your signal engine generates a trading signal:

```python
from trendcortex.ai_logger import create_strategy_log, serialize_for_ai_log

# Your signal generation code...
signal = generate_trading_signal(market_data)

# Log the strategy
log_entry = create_strategy_log(
    model="ema-crossover-v1",
    input_data=serialize_for_ai_log({
        "timeframe": "15m",
        "ema_fast": 44800,
        "ema_slow": 44600,
        "rsi": 32,
        "price": 44950,
    }),
    output_data=serialize_for_ai_log({
        "signal": "BUY",
        "confidence": 0.87,
        "entry": 44950,
        "stop_loss": 44050,
        "take_profit": 46000,
    }),
    explanation=(
        "EMA fast crossed above EMA slow with RSI oversold. "
        "Strong buy signal with 87% confidence."
    )
)

# Upload to WEEX
await ai_logger.upload_log_async(log_entry)
```

### Logging a Decision

When your ML/LLM model evaluates a signal:

```python
from trendcortex.ai_logger import create_decision_log

# Your ML/LLM evaluation code...
ml_decision = model.evaluate(signal, market_data)

# Log the decision
log_entry = create_decision_log(
    model="gpt-4-turbo-2024-12",
    input_data=serialize_for_ai_log({
        "signal": {
            "type": "BUY",
            "confidence": 0.87,
            "price": 44950,
        },
        "market_context": {
            "trend": "bullish",
            "volatility": "low",
            "support": 44500,
        },
    }),
    output_data=serialize_for_ai_log({
        "decision": "APPROVE",
        "position_size": 0.3,
        "leverage": 5,
        "confidence": 0.92,
    }),
    explanation=(
        "LLM approved trade with 92% confidence. "
        "Strong technical setup with favorable risk/reward ratio of 3.5:1."
    )
)

await ai_logger.upload_log_async(log_entry)
```

### Logging an Execution

When an order is placed and filled:

```python
from trendcortex.ai_logger import create_execution_log

# Your order execution code...
order_result = await place_order(order_params)

# Log the execution
log_entry = create_execution_log(
    model="smart-execution-v1",
    input_data=serialize_for_ai_log({
        "symbol": "BTCUSDT",
        "side": "BUY",
        "size": 0.3,
        "order_type": "LIMIT",
        "price": 44950,
    }),
    output_data=serialize_for_ai_log({
        "order_id": "987654321",
        "status": "FILLED",
        "filled_price": 44945,
        "filled_size": 0.3,
        "slippage": -5,  # Negative = favorable
    }),
    explanation=(
        "Limit order filled at 44945, saving $1.50 in slippage. "
        "Position opened: 0.3 BTC LONG with 5x leverage."
    ),
    order_id="987654321"  # Link to actual order
)

await ai_logger.upload_log_async(log_entry)
```

---

## API Reference

### Classes

#### `WEEXAILogger`

Main logger class for uploading AI logs to WEEX.

**Constructor:**
```python
WEEXAILogger(
    api_key: str,
    api_secret: str,
    api_passphrase: str,
    base_url: str = "https://api-contract.weex.com",
    timeout: int = 10
)
```

**Methods:**

- `upload_log_async(log_entry: AILogEntry) -> Dict[str, Any]`
  - Upload log asynchronously
  - Returns: Response from WEEX API
  - Raises: `aiohttp.ClientError`, `ValueError`

- `upload_log_sync(log_entry: AILogEntry) -> Dict[str, Any]`
  - Upload log synchronously
  - Returns: Response from WEEX API

#### `AILogEntry`

Data class representing a log entry.

**Attributes:**
- `stage` (str): Workflow stage
- `model` (str): Model name/version
- `input_data` (Union[Dict, str]): Model input
- `output_data` (Union[Dict, str]): Model output
- `explanation` (str): Human-readable explanation
- `order_id` (Optional[str]): Associated order ID
- `timestamp` (Optional[datetime]): Log timestamp

**Methods:**
- `to_weex_payload() -> Dict[str, Any]`: Convert to WEEX API format

#### `AILogStage`

Enum for log stages.

**Values:**
- `STRATEGY_GENERATION`: "Strategy Generation"
- `DECISION_MAKING`: "Decision Making"
- `EXECUTION`: "Execution"
- `RISK_ASSESSMENT`: "Risk Assessment"
- `SIGNAL_ANALYSIS`: "Signal Analysis"

### Helper Functions

#### `create_strategy_log()`

Create a log entry for strategy generation.

```python
create_strategy_log(
    model: str,
    input_data: Union[Dict, str],
    output_data: Union[Dict, str],
    explanation: str,
    order_id: Optional[str] = None
) -> AILogEntry
```

#### `create_decision_log()`

Create a log entry for decision making.

```python
create_decision_log(
    model: str,
    input_data: Union[Dict, str],
    output_data: Union[Dict, str],
    explanation: str,
    order_id: Optional[str] = None
) -> AILogEntry
```

#### `create_execution_log()`

Create a log entry for execution.

```python
create_execution_log(
    model: str,
    input_data: Union[Dict, str],
    output_data: Union[Dict, str],
    explanation: str,
    order_id: Optional[str] = None
) -> AILogEntry
```

#### `serialize_for_ai_log()`

Safely serialize data for logging.

```python
serialize_for_ai_log(data: Any) -> Union[Dict, str]
```

Handles:
- Dictionaries, strings (pass through)
- Pandas DataFrames (convert to dict)
- Numpy arrays (convert to list)
- Decimals (convert to float)
- Datetimes (convert to ISO string)
- Other objects (convert to string)

---

## Integration Examples

### Example 1: Complete Trading Loop with Logging

```python
import asyncio
from trendcortex.config import Config
from trendcortex.api_client import WEEXAPIClient
from trendcortex.signal_engine import SignalEngine
from trendcortex.model_integration import HybridDecisionEngine
from trendcortex.execution import TradeExecutor
from trendcortex.ai_logger import (
    WEEXAILogger,
    create_strategy_log,
    create_decision_log,
    create_execution_log,
    serialize_for_ai_log,
)

async def trading_loop():
    # Initialize components
    config = Config.load("config.json")
    api = WEEXAPIClient(config)
    signal_engine = SignalEngine(config)
    decision_engine = HybridDecisionEngine(config)
    executor = TradeExecutor(api, config)
    ai_logger = WEEXAILogger(
        api_key=config.api.key,
        api_secret=config.api.secret,
        api_passphrase=config.api.passphrase
    )
    
    symbol = "BTCUSDT"
    
    # 1. Generate signal
    market_data = await api.get_candles(symbol, "15m", 100)
    signals = await signal_engine.generate_signals(market_data)
    
    if signals:
        signal = signals[0]
        
        # Log strategy generation
        log_entry = create_strategy_log(
            model="signal-engine-v1",
            input_data=serialize_for_ai_log(market_data.tail(5)),
            output_data=serialize_for_ai_log({
                "signal": signal.direction.value,
                "confidence": signal.confidence,
            }),
            explanation=f"Generated {signal.signal_type.value} signal"
        )
        await ai_logger.upload_log_async(log_entry)
        
        # 2. AI decision
        ml_decision = await decision_engine.make_decision(
            signal, market_data, {}
        )
        
        # Log decision (already done in model_integration.py)
        # This is automatic if you use the integrated version
        
        # 3. Execute trade
        if ml_decision.approve_trade:
            result = await executor.execute_signal(
                signal, risk_assessment, contract_specs
            )
            
            # Log execution (already done in execution.py)
            # This is automatic if you use the integrated version

asyncio.run(trading_loop())
```

### Example 2: Custom Model with Logging

```python
from trendcortex.ai_logger import create_decision_log, serialize_for_ai_log

class CustomMLModel:
    def __init__(self, config):
        self.config = config
        self.ai_logger = WEEXAILogger(
            api_key=config.api.key,
            api_secret=config.api.secret,
            api_passphrase=config.api.passphrase
        )
    
    async def predict(self, features):
        # Your model inference
        prediction = self.model.predict(features)
        confidence = float(prediction[0])
        
        # Log the prediction
        log_entry = create_decision_log(
            model="custom-xgboost-v2.1",
            input_data=serialize_for_ai_log(features),
            output_data=serialize_for_ai_log({
                "prediction": "BUY" if confidence > 0.5 else "SELL",
                "confidence": confidence,
                "probability": confidence,
            }),
            explanation=f"XGBoost model predicted with {confidence:.1%} confidence"
        )
        
        try:
            await self.ai_logger.upload_log_async(log_entry)
        except Exception as e:
            print(f"Failed to log: {e}")
        
        return prediction
```

### Example 3: Batch Logging

If you need to log multiple entries at once:

```python
async def batch_log_trades(trades):
    ai_logger = WEEXAILogger(...)
    
    for trade in trades:
        log_entry = create_execution_log(
            model="backtester-v1",
            input_data=serialize_for_ai_log(trade.params),
            output_data=serialize_for_ai_log(trade.result),
            explanation=trade.explanation,
            order_id=trade.order_id
        )
        
        try:
            await ai_logger.upload_log_async(log_entry)
            await asyncio.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Failed to log trade {trade.order_id}: {e}")
```

---

## Troubleshooting

### Error: "Failed to upload AI log: 401"

**Cause**: Invalid API credentials

**Solution**:
1. Check `config.json` has correct `api.key`, `api.secret`, `api.passphrase`
2. Verify credentials are for the correct WEEX account
3. Ensure passphrase matches exactly (case-sensitive)

### Error: "Failed to upload AI log: 400"

**Cause**: Invalid payload format

**Solution**:
1. Check that all required fields are present
2. Ensure `stage` is one of the valid stages
3. Verify data is properly serialized with `serialize_for_ai_log()`

### Error: "WEEX API error: code=30015"

**Cause**: Rate limit exceeded

**Solution**:
1. Add delays between log uploads
2. Batch logs and upload less frequently
3. Check WEEX API rate limits

### Error: "Network error uploading AI log"

**Cause**: Connection issues

**Solution**:
1. Check internet connection
2. Verify WEEX API is accessible
3. Check firewall settings
4. Try increasing timeout value

### Data Serialization Issues

If you get serialization errors:

```python
# Bad: Passing complex objects
log_entry = create_decision_log(
    input_data=pandas_dataframe  # May cause issues
)

# Good: Use serialize_for_ai_log()
log_entry = create_decision_log(
    input_data=serialize_for_ai_log(pandas_dataframe)  # Safe
)
```

### Missing Logs

If logs aren't appearing in WEEX dashboard:

1. **Check response codes**: Look for `{"code": "0", "msg": "success"}`
2. **Verify order_id**: Ensure order_id matches actual WEEX orders
3. **Check timing**: Logs may take a few minutes to appear
4. **Review stage names**: Must be exact match to WEEX requirements

---

## Best Practices

### 1. Always Use Helper Functions

```python
# Good
log = create_decision_log(model, input, output, explanation)

# Avoid
log = AILogEntry(
    stage="Decision Making",  # Easy to typo
    model=model,
    input_data=input,
    output_data=output,
    explanation=explanation
)
```

### 2. Serialize Complex Data

```python
# Always serialize DataFrames, arrays, etc.
input_data = serialize_for_ai_log(market_data)
```

### 3. Provide Detailed Explanations

```python
# Bad
explanation = "Trade approved"

# Good
explanation = (
    "LLM approved BUY signal with 92% confidence. "
    "Technical indicators show oversold conditions with bullish divergence. "
    "Risk-reward ratio of 3.5:1 justifies position size of 0.3 BTC with 5x leverage. "
    "Stop loss at 44050 provides 2% downside protection."
)
```

### 4. Handle Errors Gracefully

```python
try:
    await ai_logger.upload_log_async(log_entry)
except Exception as e:
    logger.error(f"Failed to upload log: {e}")
    # Continue trading - don't let logging errors stop the bot
```

### 5. Link Orders to Logs

```python
# Always pass order_id when available
log_entry = create_execution_log(
    ...,
    order_id=order_result.order_id  # Important!
)
```

---

## Competition Requirements

For WEEX AI Wars, ensure you:

1. ✅ **Log all AI decisions**: Every trade must have logs
2. ✅ **Include all stages**: Strategy → Decision → Execution
3. ✅ **Link to orders**: Use `order_id` parameter
4. ✅ **Provide explanations**: Human-readable reasoning
5. ✅ **Use correct stages**: Match WEEX stage names exactly
6. ✅ **Handle errors**: Don't let logging failures stop trading

---

## Additional Resources

- [WEEX API Documentation](https://www.weex.com/api-doc/ai/intro)
- [AI Wars Competition Rules](https://www.weex.com/events/ai-trading)
- [TrendCortex Main Documentation](../README.md)
- [Configuration Guide](../QUICKSTART.md)

---

**Last Updated**: December 26, 2024  
**Module**: `trendcortex/ai_logger.py`  
**Version**: 1.0.0
