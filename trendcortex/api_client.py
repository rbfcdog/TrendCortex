"""
WEEX API Client Module

Handles async communication with WEEX exchange API including:
- REST API calls with HMAC-SHA256 authentication
- WebSocket connections for real-time data
- Rate limiting and retry logic
- Error handling and response validation
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from trendcortex.config import Config
from trendcortex.logger import get_logger


class WEEXAPIError(Exception):
    """Base exception for WEEX API errors."""
    def __init__(self, code: int, message: str, status_code: int = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"WEEX API Error {code}: {message}")


class WEEXAPIClient:
    """
    Async client for WEEX exchange API.
    
    Supports both REST API and WebSocket connections with proper authentication,
    rate limiting, and error handling.
    """
    
    def __init__(self, config: Config):
        """
        Initialize WEEX API client.
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.logger = get_logger()
        
        self.base_url = config.api.base_url
        self.api_key = config.api.key
        self.secret_key = config.api.secret
        self.passphrase = config.api.passphrase
        self.locale = config.api.locale
        self.timeout = config.api.timeout
        self.max_retries = config.api.max_retries
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Rate limiting
        self.rate_limit_delay = config.api.rate_limit_delay
        self.last_request_time = 0.0
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Initialize HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info("WEEX API client connected")
    
    async def close(self) -> None:
        """Close HTTP session and WebSocket connections."""
        if self.ws:
            await self.ws.close()
            self.ws = None
        
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("WEEX API client disconnected")
    
    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str = "",
        body: str = "",
    ) -> str:
        """
        Generate HMAC-SHA256 signature for authenticated requests.
        
        Args:
            timestamp: Request timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            query_string: URL query parameters
            body: Request body (for POST requests)
            
        Returns:
            Base64-encoded signature
        """
        message = timestamp + method.upper() + request_path + query_string + body
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _get_headers(
        self,
        method: str,
        request_path: str,
        query_string: str = "",
        body: str = "",
    ) -> Dict[str, str]:
        """
        Generate request headers with authentication.
        
        Args:
            method: HTTP method
            request_path: API endpoint path
            query_string: URL query parameters
            body: Request body
            
        Returns:
            Headers dictionary
        """
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(
            timestamp, method, request_path, query_string, body
        )
        
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": self.locale,
        }
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        authenticated: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute HTTP request to WEEX API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: URL query parameters
            data: Request body data
            authenticated: Whether request requires authentication
            
        Returns:
            Response data as dictionary
            
        Raises:
            WEEXAPIError: If API returns error
        """
        if self.session is None:
            await self.connect()
        
        await self._rate_limit()
        
        # Prepare request
        url = f"{self.base_url}{endpoint}"
        query_string = ""
        body_str = ""
        
        if params:
            query_string = "?" + urlencode(params)
            url += query_string
        
        if data:
            body_str = json.dumps(data)
        
        # Generate headers
        if authenticated:
            headers = self._get_headers(method, endpoint, query_string, body_str)
        else:
            headers = {"Content-Type": "application/json"}
        
        # Execute request
        async with self.session.request(
            method,
            url,
            headers=headers,
            data=body_str if body_str else None,
        ) as response:
            response_text = await response.text()
            
            # Handle HTTP errors
            if response.status != 200:
                self.logger.error(
                    f"API request failed: {method} {endpoint}",
                    status=response.status,
                    response=response_text,
                )
                raise WEEXAPIError(
                    code=response.status,
                    message=response_text,
                    status_code=response.status,
                )
            
            # Parse response
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON response: {response_text}")
                raise WEEXAPIError(
                    code=-1,
                    message="Invalid JSON response",
                    status_code=response.status,
                )
            
            # Check for API errors
            if isinstance(response_data, dict):
                if "code" in response_data and response_data["code"] != "200":
                    error_msg = response_data.get("msg", "Unknown error")
                    raise WEEXAPIError(
                        code=int(response_data["code"]),
                        message=error_msg,
                    )
            
            return response_data
    
    # ==================== Public API Methods ====================
    
    async def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time.
        
        Returns:
            Server time information
        """
        return await self._request("GET", "/capi/v2/market/time", authenticated=False)
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "cmt_btcusdt")
            
        Returns:
            Ticker data including prices, volume, etc.
        """
        params = {"symbol": symbol}
        return await self._request("GET", "/capi/v2/market/ticker", params=params, authenticated=False)
    
    async def get_tickers(self) -> List[Dict[str, Any]]:
        """
        Get tickers for all symbols.
        
        Returns:
            List of ticker data
        """
        result = await self._request("GET", "/capi/v2/market/tickers", authenticated=False)
        return result if isinstance(result, list) else []
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """
        Get orderbook depth.
        
        Args:
            symbol: Trading pair symbol
            depth: Number of orders to fetch (default: 20)
            
        Returns:
            Orderbook data with bids and asks
        """
        params = {"symbol": symbol, "depth": depth}
        return await self._request("GET", "/capi/v2/market/depth", params=params, authenticated=False)
    
    async def get_candles(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair symbol
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles to fetch
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of candle data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        result = await self._request("GET", "/capi/v2/market/klines", params=params, authenticated=False)
        return result if isinstance(result, list) else []
    
    async def get_contract_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get contract information (precision, limits, leverage, etc.).
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Contract specification data
        """
        params = {"symbol": symbol}
        result = await self._request("GET", "/capi/v2/market/contracts", params=params, authenticated=False)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return result
    
    # ==================== Private API Methods ====================
    
    async def get_account_balance(self) -> List[Dict[str, Any]]:
        """
        Get account balance information.
        
        Returns:
            List of asset balances
        """
        result = await self._request("GET", "/capi/v2/account/assets")
        return result if isinstance(result, list) else []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open positions
        """
        params = {"symbol": symbol} if symbol else {}
        result = await self._request("GET", "/capi/v2/account/positions", params=params)
        return result if isinstance(result, list) else []
    
    async def set_leverage(
        self,
        symbol: str,
        margin_mode: int,
        long_leverage: int,
        short_leverage: int,
    ) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading pair symbol
            margin_mode: Margin mode (1=cross, 2=isolated)
            long_leverage: Leverage for long positions
            short_leverage: Leverage for short positions
            
        Returns:
            Success response
        """
        data = {
            "symbol": symbol,
            "marginMode": margin_mode,
            "longLeverage": str(long_leverage),
            "shortLeverage": str(short_leverage),
        }
        return await self._request("POST", "/capi/v2/account/leverage", data=data)
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        size: str,
        order_type: str = "1",
        price: Optional[str] = None,
        client_oid: Optional[str] = None,
        position_side: str = "1",
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (1=buy/open_long, 2=sell/open_short, 3=close_long, 4=close_short)
            size: Order size
            order_type: Order type (1=limit, 2=market, 3=post_only, 4=fok, 5=ioc)
            price: Order price (required for limit orders)
            client_oid: Client order ID (optional)
            position_side: Position side (1=long, 2=short)
            reduce_only: Whether this is a reduce-only order
            
        Returns:
            Order response with order_id
        """
        data = {
            "symbol": symbol,
            "size": size,
            "type": side,
            "order_type": order_type,
            "match_price": "0",
        }
        
        if price:
            data["price"] = price
        if client_oid:
            data["client_oid"] = client_oid
        if reduce_only:
            data["reduce_only"] = "1"
        
        return await self._request("POST", "/capi/v2/order/placeOrder", data=data)
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
        """
        data = {
            "symbol": symbol,
            "orderId": order_id,
        }
        return await self._request("POST", "/capi/v2/order/cancelOrder", data=data)
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Batch cancellation response
        """
        data = {"symbol": symbol}
        return await self._request("POST", "/capi/v2/order/cancelAllOrders", data=data)
    
    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Get order details.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            
        Returns:
            Order details
        """
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("GET", "/capi/v2/order/detail", params=params)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of open orders
        """
        params = {"symbol": symbol} if symbol else {}
        result = await self._request("GET", "/capi/v2/order/openOrders", params=params)
        return result.get("list", []) if isinstance(result, dict) else []
    
    async def get_trade_fills(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get trade fill history.
        
        Args:
            symbol: Trading pair symbol
            order_id: Filter by order ID (optional)
            limit: Number of records to fetch
            
        Returns:
            List of trade fills
        """
        params = {"symbol": symbol, "limit": limit}
        if order_id:
            params["orderId"] = order_id
        
        result = await self._request("GET", "/capi/v2/order/fills", params=params)
        return result.get("list", []) if isinstance(result, dict) else []
    
    # ==================== WebSocket Methods ====================
    # TODO: Implement WebSocket connection for real-time data streams
    # 
    # async def connect_websocket(self) -> None:
    #     """Establish WebSocket connection for real-time market data."""
    #     pass
    # 
    # async def subscribe_ticker(self, symbol: str) -> None:
    #     """Subscribe to ticker updates."""
    #     pass
    # 
    # async def subscribe_orderbook(self, symbol: str) -> None:
    #     """Subscribe to orderbook updates."""
    #     pass
    # 
    # async def subscribe_trades(self, symbol: str) -> None:
    #     """Subscribe to trade updates."""
    #     pass
