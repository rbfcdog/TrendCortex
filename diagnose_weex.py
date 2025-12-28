#!/usr/bin/env python3
"""
Simple Weex API Diagnostic Script

Tests basic connectivity to Weex API endpoints.
"""

import asyncio
import aiohttp
import time
import hmac
import hashlib
import base64


async def test_api():
    """Test basic Weex API connectivity."""
    
    # Your API credentials
    API_KEY = "weex_dd8ad364e911f203e4f00ca7c06339d6"
    SECRET_KEY = "6af415251b58ae01ff25daf053008e860f49cf7a43bad2a9d7fbb2e22d97b78d"
    PASSPHRASE = "weex6192461"
    BASE_URL = "https://api-contract.weex.com"
    
    print("=" * 60)
    print("🔍 Weex API Diagnostic Test")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY[:20]}...")
    print()
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Public endpoint (no auth)
        print("📊 Test 1: Public endpoint - Server Time")
        try:
            async with session.get(f"{BASE_URL}/capi/v2/market/time") as resp:
                print(f"Status: {resp.status}")
                text = await resp.text()
                print(f"Response: {text[:200]}")
                print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
        
        # Test 2: Another public endpoint
        print("📊 Test 2: Public endpoint - Ticker")
        try:
            async with session.get(f"{BASE_URL}/capi/v2/market/ticker?symbol=cmt_btcusdt") as resp:
                print(f"Status: {resp.status}")
                text = await resp.text()
                print(f"Response: {text[:200]}")
                print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
        
        # Test 3: Authenticated endpoint
        print("💰 Test 3: Authenticated endpoint - Account Balance")
        try:
            # Generate signature
            timestamp = str(int(time.time() * 1000))
            method = "GET"
            request_path = "/capi/v2/account/assets"
            
            message = timestamp + method + request_path
            signature = hmac.new(
                SECRET_KEY.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            signature_b64 = base64.b64encode(signature).decode()
            
            headers = {
                "ACCESS-KEY": API_KEY,
                "ACCESS-SIGN": signature_b64,
                "ACCESS-TIMESTAMP": timestamp,
                "ACCESS-PASSPHRASE": PASSPHRASE,
                "Content-Type": "application/json",
                "locale": "en-US"
            }
            
            async with session.get(f"{BASE_URL}{request_path}", headers=headers) as resp:
                print(f"Status: {resp.status}")
                text = await resp.text()
                print(f"Response: {text[:500]}")
                print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
    
    print("=" * 60)
    print("✅ Diagnostic Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_api())
