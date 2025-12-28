#!/usr/bin/env python3
"""
Quick API Connection Test Script

Tests your WEEX API configuration by making a simple request.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from trendcortex.config import Config
from trendcortex.api_client import WEEXAPIClient


async def test_connection():
    """Test WEEX API connection."""
    print("=" * 60)
    print("TrendCortex API Connection Test")
    print("=" * 60)
    print()
    
    try:
        # Load config
        print("Loading configuration...")
        config = Config.load("config.json")
        print(f"✅ Config loaded")
        print(f"   API URL: {config.api.base_url}")
        print(f"   API Key: {config.api.key[:8]}...")
        print()
        
        # Validate credentials
        if config.api.key == "YOUR_API_KEY_HERE":
            print("❌ Error: API credentials not configured!")
            print("   Please edit config.json and add your WEEX API credentials")
            return False
        
        # Create client
        print("Connecting to WEEX API...")
        async with WEEXAPIClient(config) as client:
            
            # Test 1: Server time
            print("\n📡 Test 1: Fetching server time...")
            server_time = await client.get_server_time()
            print(f"✅ Server time: {server_time['iso']}")
            
            # Test 2: Account balance
            print("\n💰 Test 2: Fetching account balance...")
            balances = await client.get_account_balance()
            
            if balances:
                print("✅ Account balance retrieved:")
                for balance in balances:
                    coin = balance.get('coinName')
                    available = balance.get('available')
                    print(f"   {coin}: {available}")
            else:
                print("⚠️  No balances found (account may be empty)")
            
            # Test 3: Market data
            print("\n📊 Test 3: Fetching market data...")
            symbol = config.trading.primary_symbol
            ticker = await client.get_ticker(symbol)
            print(f"✅ {symbol} price: ${ticker['last']}")
            
            # Test 4: Contract info
            print("\n📋 Test 4: Fetching contract information...")
            contract = await client.get_contract_info(symbol)
            print(f"✅ Contract info for {symbol}:")
            print(f"   Min size: {contract['min_order_size']}")
            print(f"   Max leverage: {contract['max_leverage']}x")
            print(f"   Maker fee: {float(contract['maker_fee'])*100:.3f}%")
            print(f"   Taker fee: {float(contract['taker_fee'])*100:.3f}%")
            
        print()
        print("=" * 60)
        print("✅ All API tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Review configuration in config.json")
        print("  2. Run dry-run test: python3 main.py --dry-run")
        print("  3. Monitor logs in logs/ directory")
        print()
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ API test failed!")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        print()
        print("Common issues:")
        print("  • Check API credentials in config.json")
        print("  • Ensure IP is whitelisted with WEEX")
        print("  • Verify system time is synchronized")
        print("  • Check internet connection")
        print()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
