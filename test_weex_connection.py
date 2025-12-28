#!/usr/bin/env python3
"""
Test Weex API Connection

Quick script to verify API credentials are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trendcortex.config import Config
from trendcortex.api_client import WEEXAPIClient
from trendcortex.logger import setup_logging


async def test_api_connection():
    """Test basic API connectivity and authentication."""
    
    print("=" * 60)
    print("🔧 Testing Weex API Connection")
    print("=" * 60)
    print()
    
    try:
        # Load config
        print("📋 Loading configuration from config.json...")
        config = Config.load("config.json")
        print(f"✅ Config loaded")
        print(f"   API Key: {config.api.key[:20]}...")
        print(f"   Base URL: {config.api.base_url}")
        print()
        
        # Setup logging
        setup_logging(
            log_dir=config.logging.log_directory,
            level=config.logging.level,
            console_output=config.logging.console_output,
            json_format=config.logging.json_format
        )
        
        # Initialize API client
        print("🔌 Connecting to Weex API...")
        async with WEEXAPIClient(config) as client:
            print("✅ Connected!")
            print()
            
            # Test 1: Get server time
            print("⏰ Test 1: Getting server time...")
            try:
                server_time = await client.get_server_time()
                print(f"✅ Server time: {server_time}")
                print()
            except Exception as e:
                print(f"❌ Failed to get server time: {e}")
                print()
            
            # Test 2: Get account balance
            print("💰 Test 2: Getting account balance...")
            try:
                balance = await client.get_account_balance()
                print(f"✅ Account balance retrieved successfully!")
                print(f"   Response: {balance}")
                print()
            except Exception as e:
                print(f"❌ Failed to get account balance: {e}")
                print()
            
            # Test 3: Get market data (public endpoint - no auth needed)
            print("📊 Test 3: Getting market data for BTC/USDT...")
            try:
                ticker = await client.get_ticker("cmt_btcusdt")
                print(f"✅ Market data retrieved!")
                if 'data' in ticker:
                    data = ticker['data']
                    print(f"   Symbol: {data.get('symbol', 'N/A')}")
                    print(f"   Last Price: {data.get('last', 'N/A')}")
                    print(f"   24h Volume: {data.get('volume24h', 'N/A')}")
                else:
                    print(f"   Response: {ticker}")
                print()
            except Exception as e:
                print(f"❌ Failed to get market data: {e}")
                print()
            
        print("=" * 60)
        print("✅ API Connection Test Complete!")
        print("=" * 60)
        
    except FileNotFoundError:
        print("❌ Error: config.json not found!")
        print("   Please make sure config.json exists in the project root.")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_api_connection())
    sys.exit(0 if success else 1)
