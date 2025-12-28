#!/usr/bin/env python3
"""
WEEX AI Logger Test Script

This script demonstrates how to use the WEEX AI logging functionality.
Run this after you've configured your API credentials in config.json.

Usage:
    python test_ai_logger.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trendcortex.config import Config
from trendcortex.ai_logger import (
    WEEXAILogger,
    create_strategy_log,
    create_decision_log,
    create_execution_log,
    serialize_for_ai_log,
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_strategy_log(ai_logger: WEEXAILogger):
    """Test logging a strategy generation event"""
    print_section("Test 1: Strategy Generation Log")
    
    # Simulate a strategy generation
    log_entry = create_strategy_log(
        model="ema-crossover-v1.0",
        input_data=serialize_for_ai_log({
            "timeframe": "15m",
            "indicators": {
                "ema_fast": 44800,
                "ema_slow": 44600,
                "rsi": 32,
                "volume_ratio": 1.5,
            },
            "current_price": 44950,
        }),
        output_data=serialize_for_ai_log({
            "signal": "BUY",
            "confidence": 0.87,
            "entry_price": 44950,
            "stop_loss": 44050,
            "take_profit": 46000,
            "strategy_name": "EMA Crossover + RSI Oversold",
        }),
        explanation=(
            "EMA fast (44800) crossed above EMA slow (44600) with RSI at oversold level (32). "
            "Volume is 1.5x average, confirming strong buying pressure. "
            "Generated BUY signal with 87% confidence."
        )
    )
    
    print("Log Entry Created:")
    print(f"  Stage: {log_entry.stage}")
    print(f"  Model: {log_entry.model}")
    print(f"  Explanation: {log_entry.explanation}")
    
    try:
        print("\nUploading to WEEX...")
        response = await ai_logger.upload_log_async(log_entry)
        print("✅ Success!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def test_decision_log(ai_logger: WEEXAILogger):
    """Test logging a decision making event"""
    print_section("Test 2: Decision Making Log")
    
    # Simulate an LLM decision
    log_entry = create_decision_log(
        model="gpt-4-turbo-2024-12",
        input_data=serialize_for_ai_log({
            "signal": {
                "type": "BUY",
                "symbol": "BTCUSDT",
                "confidence": 0.87,
                "price": 44950,
            },
            "market_context": {
                "trend": "bullish",
                "volatility": "low",
                "support_level": 44500,
                "resistance_level": 46000,
            },
            "risk_metrics": {
                "account_balance": 10000,
                "max_position_size": 0.5,
                "current_exposure": 0.2,
            }
        }),
        output_data=serialize_for_ai_log({
            "decision": "APPROVE",
            "position_size": 0.3,
            "leverage": 5,
            "confidence_adjustment": 0.92,
            "risk_reward_ratio": 3.5,
            "reasoning": "Strong technical setup with favorable risk/reward",
        }),
        explanation=(
            "LLM approved BUY signal with adjusted confidence of 92%. "
            "Technical indicators show strong oversold conditions with bullish divergence. "
            "Risk-reward ratio of 3.5:1 justifies position size of 0.3 BTC with 5x leverage. "
            "Stop loss at 44050 provides 2% downside protection."
        )
    )
    
    print("Log Entry Created:")
    print(f"  Stage: {log_entry.stage}")
    print(f"  Model: {log_entry.model}")
    print(f"  Explanation: {log_entry.explanation[:100]}...")
    
    try:
        print("\nUploading to WEEX...")
        response = await ai_logger.upload_log_async(log_entry)
        print("✅ Success!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def test_execution_log(ai_logger: WEEXAILogger):
    """Test logging an execution event"""
    print_section("Test 3: Execution Log")
    
    # Simulate an order execution
    log_entry = create_execution_log(
        model="smart-execution-v1.0",
        input_data=serialize_for_ai_log({
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "LIMIT",
            "price": 44950,
            "size": 0.3,
            "leverage": 5,
            "time_in_force": "GTC",
        }),
        output_data=serialize_for_ai_log({
            "order_id": "test_" + str(asyncio.get_event_loop().time()),
            "status": "FILLED",
            "filled_price": 44945,
            "filled_size": 0.3,
            "fill_time": "2024-12-26T10:30:00Z",
            "commission": 0.0002,
            "slippage": -5,
        }),
        explanation=(
            "Limit order placed at 44950 and filled at 44945, saving $1.50 in slippage. "
            "Order fully filled within 2 seconds. "
            "Position now open: 0.3 BTC LONG with 5x leverage at 44945. "
            "Stop loss set at 44050, take profit at 46000."
        ),
        order_id="test_order_12345"
    )
    
    print("Log Entry Created:")
    print(f"  Stage: {log_entry.stage}")
    print(f"  Model: {log_entry.model}")
    print(f"  Order ID: {log_entry.order_id}")
    print(f"  Explanation: {log_entry.explanation[:100]}...")
    
    try:
        print("\nUploading to WEEX...")
        response = await ai_logger.upload_log_async(log_entry)
        print("✅ Success!")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


async def main():
    """Main test function"""
    print("=" * 80)
    print("  WEEX AI Logger Test Script")
    print("  TrendCortex - AI Wars Competition")
    print("=" * 80)
    
    # Load configuration
    print("\n📋 Loading configuration...")
    try:
        config = Config.load("config.json")
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Failed to load config.json: {e}")
        print("\nMake sure you have:")
        print("1. Created config.json from config.example.json")
        print("2. Added your WEEX API credentials")
        return
    
    # Verify credentials
    if not config.api.key or config.api.key == "your_weex_api_key":
        print("\n⚠️  Warning: API credentials not configured!")
        print("Please edit config.json and add your WEEX API credentials:")
        print("  - api.key")
        print("  - api.secret")
        print("  - api.passphrase")
        return
    
    print(f"API Key: {config.api.key[:8]}..." + "*" * 20)
    
    # Initialize AI logger
    print("\n🔧 Initializing WEEX AI Logger...")
    ai_logger = WEEXAILogger(
        api_key=config.api.key,
        api_secret=config.api.secret,
        api_passphrase=config.api.passphrase,
        base_url=config.api.base_url,
    )
    print("✅ AI Logger initialized")
    
    # Run tests
    results = []
    
    # Test 1: Strategy log
    result1 = await test_strategy_log(ai_logger)
    results.append(("Strategy Generation", result1))
    await asyncio.sleep(1)  # Rate limiting
    
    # Test 2: Decision log
    result2 = await test_decision_log(ai_logger)
    results.append(("Decision Making", result2))
    await asyncio.sleep(1)  # Rate limiting
    
    # Test 3: Execution log
    result3 = await test_execution_log(ai_logger)
    results.append(("Execution", result3))
    
    # Print summary
    print_section("Test Summary")
    
    all_passed = all(result for _, result in results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests passed! AI logging is working correctly.")
        print("\nYou can now:")
        print("1. Check WEEX dashboard for your AI logs")
        print("2. Integrate AI logging into your trading bot")
        print("3. See docs/AI_LOGGING.md for more examples")
    else:
        print("⚠️  Some tests failed. Common issues:")
        print("1. Invalid API credentials")
        print("2. Network connectivity issues")
        print("3. WEEX API rate limits")
        print("4. API endpoint not accessible")
        print("\nCheck the error messages above for details.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
