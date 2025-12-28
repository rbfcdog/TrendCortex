# TrendCortex Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

### Step 2: Configure API Keys

Edit `config.json`:

```json
{
  "api": {
    "key": "YOUR_WEEX_API_KEY",
    "secret": "YOUR_WEEX_SECRET_KEY",
    "passphrase": "YOUR_WEEX_PASSPHRASE"
  }
}
```

### Step 3: Test Connection

```bash
python3 << EOF
import asyncio
from trendcortex.api_client import WEEXAPIClient
from trendcortex.config import Config

async def test():
    config = Config.load()
    async with WEEXAPIClient(config) as client:
        time = await client.get_server_time()
        print(f"✅ Connected! Server time: {time['iso']}")

asyncio.run(test())
EOF
```

### Step 4: Run Dry-Run Test

```bash
# Test for 5 minutes in dry-run mode
python3 main.py --dry-run
```

Press `Ctrl+C` to stop.

### Step 5: Monitor Logs

```bash
# In another terminal, watch signal logs
tail -f logs/signals/signals_*.json | jq '.'

# Watch decision logs
tail -f logs/decisions/decisions_*.json | jq '.'

# Watch execution logs
tail -f logs/executions/executions_*.json | jq '.'
```

## 📊 Understanding the System

### Trading Flow

```
1. Data Fetch → 2. Signal Generation → 3. AI Decision → 4. Risk Check → 5. Execution
```

### Key Components

- **Signal Engine**: Generates trading signals from technical indicators
- **ML Model**: Evaluates signal quality (placeholder - add your model)
- **LLM Gate**: Optional human-like reasoning layer
- **Risk Controller**: Enforces position sizing and safety rules
- **Executor**: Places orders on exchange

## 🎯 Competition Mode (WEEX AI Wars)

### Important Rules

1. **Max Leverage**: 20x (system enforces this)
2. **Allowed Pairs**: Only 8 CMT pairs (configured in config.json)
3. **Initial Balance**: 1,000 USDT
4. **Testing**: Until Jan 5, 2026
5. **Competition**: Late February 2026

### Pre-Competition Checklist

- [ ] API credentials configured
- [ ] Tested in dry-run mode
- [ ] Signals generating correctly
- [ ] Risk limits configured (max 20x leverage)
- [ ] Only trading allowed pairs
- [ ] Logs working properly
- [ ] Understand cooldown periods

## 🧪 Testing Your Strategy

### Run Backtest (TODO)

```bash
python3 main.py --backtest --start-date 2025-01-01 --end-date 2025-01-31
```

### Monitor Performance

Check logs for metrics:
```bash
grep "performance" logs/executions/*.json | jq '.'
```

## 🔧 Customization

### Add Your ML Model

Edit `trendcortex/model_integration.py`:

```python
def train(self, training_data, labels):
    # Your training code here
    from sklearn.ensemble import RandomForestClassifier
    self.model = RandomForestClassifier(n_estimators=100)
    self.model.fit(training_data[self.features], labels)
```

### Add Your LLM

Edit `trendcortex/model_integration.py`:

```python
async def make_decision(self, signal, ml_decision, market_context):
    # Your LLM integration here
    import openai
    response = await openai.ChatCompletion.create(...)
```

### Customize Signals

Edit `trendcortex/signal_engine.py` to add new signal types.

### Adjust Risk Parameters

Edit `config.json`:

```json
{
  "risk": {
    "max_leverage": 10,
    "stop_loss_percent": 1.5,
    "cooldown_seconds": 600
  }
}
```

## 🐛 Troubleshooting

### "521 Web Server is Down"
Your IP is not whitelisted. Contact WEEX support.

### "Signature Verification Failed"
- Check API credentials in config.json
- Ensure system time is synchronized (use NTP)
- Verify passphrase is correct

### No Signals Generated
- Check that you have enough historical data
- Verify indicators are calculating (check logs)
- Lower `min_confidence` in config.json temporarily

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📚 Further Reading

- [Full Documentation](README.md)
- [WEEX API Docs](https://www.weex.com/api-doc/ai/intro)
- [Competition Rules](https://www.weex.com/events/ai-trading)

## 🆘 Getting Help

1. Check logs in `logs/` directory
2. Enable debug logging in config.json
3. Test components individually (see tests/)
4. Contact WEEX support via Telegram

---

**Good luck in the competition! 🏆**
