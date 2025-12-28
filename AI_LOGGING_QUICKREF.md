# WEEX AI Logging - Quick Reference Card

## 🚀 Quick Start

```bash
# 1. Test AI Logger
./test_ai_logger.py

# 2. Run bot with AI logging (dry-run)
python main.py --dry-run

# 3. Check logs in WEEX dashboard
```

## 📝 Basic Usage

### Import
```python
from trendcortex.ai_logger import (
    WEEXAILogger,
    create_strategy_log,
    create_decision_log,
    create_execution_log,
    serialize_for_ai_log,
)
```

### Initialize
```python
ai_logger = WEEXAILogger(
    api_key="your_key",
    api_secret="your_secret",
    api_passphrase="your_passphrase"
)
```

### Log Strategy
```python
log = create_strategy_log(
    model="model-name-v1",
    input_data=serialize_for_ai_log(market_data),
    output_data=serialize_for_ai_log(signal),
    explanation="Why this strategy was chosen"
)
await ai_logger.upload_log_async(log)
```

### Log Decision
```python
log = create_decision_log(
    model="gpt-4-turbo",
    input_data=serialize_for_ai_log(signal_and_context),
    output_data=serialize_for_ai_log(decision),
    explanation="Why approved/rejected"
)
await ai_logger.upload_log_async(log)
```

### Log Execution
```python
log = create_execution_log(
    model="execution-engine-v1",
    input_data=serialize_for_ai_log(order_params),
    output_data=serialize_for_ai_log(execution_result),
    explanation="How execution was performed",
    order_id="actual_order_id"
)
await ai_logger.upload_log_async(log)
```

## 🎯 Three Log Stages

| Stage | When to Use | Module |
|-------|-------------|--------|
| **Strategy Generation** | After generating trading signal | `signal_engine.py` |
| **Decision Making** | After ML/LLM evaluates signal | `model_integration.py` |
| **Execution** | After placing/filling order | `execution.py` |

## 🔧 Helper Functions

### `serialize_for_ai_log(data)`
Safely serialize any data type:
- DataFrames → dict
- NumPy arrays → list
- Decimal → float
- Datetime → ISO string
- Objects → string

### Stage Creators
- `create_strategy_log()` - For signal generation
- `create_decision_log()` - For ML/LLM decisions
- `create_execution_log()` - For order execution
- `create_risk_assessment_log()` - For risk evaluation

## ⚠️ Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| 401 | Invalid credentials | Check config.json |
| 400 | Invalid payload | Use serialize_for_ai_log() |
| 30015 | Rate limit | Add delays between logs |
| Network error | Connection issue | Check internet/firewall |

## ✅ Automatic Logging

**Already integrated in:**
- ✅ `model_integration.py` - ML evaluation logs
- ✅ `model_integration.py` - LLM decision logs
- ✅ `execution.py` - Execution logs

**No manual logging needed for standard trading flow!**

## 📋 Checklist

Before going live:
- [ ] API credentials configured in config.json
- [ ] Run `./test_ai_logger.py` - all tests pass
- [ ] Test dry-run: `python main.py --dry-run`
- [ ] Check WEEX dashboard for logs
- [ ] Review `docs/AI_LOGGING.md` for details

## 🆘 Need Help?

1. **Full Documentation**: `docs/AI_LOGGING.md`
2. **Implementation Details**: `AI_LOGGING_IMPLEMENTATION.md`
3. **Test Script**: `./test_ai_logger.py`
4. **Examples**: See documentation for 10+ examples

## 📞 API Endpoint

```
POST https://api-contract.weex.com/capi/v2/order/uploadAiLog

Headers:
  ACCESS-KEY: your_api_key
  ACCESS-SIGN: hmac_sha256_signature
  ACCESS-TIMESTAMP: milliseconds
  ACCESS-PASSPHRASE: your_passphrase
  Content-Type: application/json

Body:
{
  "orderId": "optional",
  "stage": "Strategy Generation | Decision Making | Execution",
  "model": "model-name-v1",
  "input": "serialized_input",
  "output": "serialized_output",
  "explanation": "human_readable_text"
}
```

## 🎯 Competition Rules

✅ Log all AI decisions  
✅ Include all three stages  
✅ Link to order IDs  
✅ Provide explanations  
✅ Don't stop on logging errors

---

**Version**: 1.0.0  
**Last Updated**: December 26, 2024  
**Module**: `trendcortex/ai_logger.py`
