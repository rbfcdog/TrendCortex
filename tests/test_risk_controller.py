"""
Unit tests for Risk Controller module.
"""

import pytest
from datetime import datetime

from trendcortex.config import Config
from trendcortex.risk_controller import RiskController, RiskAssessment
from trendcortex.signal_engine import TradingSignal, SignalType, SignalDirection


class TestRiskController:
    """Test suite for RiskController"""
    
    @pytest.fixture
    def config(self):
        """Load test configuration"""
        return Config.load("config.example.json")
    
    @pytest.fixture
    def risk_controller(self, config):
        """Create RiskController instance"""
        return RiskController(config)
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample trading signal"""
        return TradingSignal(
            timestamp=datetime.now(),
            symbol="cmt_btcusdt",
            signal_type=SignalType.EMA_CROSS,
            direction=SignalDirection.LONG,
            confidence=0.8,
            price=50000.0,
            indicators={},
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
        )
    
    @pytest.mark.asyncio
    async def test_validate_trade_with_sufficient_balance(self, risk_controller, sample_signal):
        """Test trade validation with sufficient balance"""
        result = await risk_controller.validate_trade(
            signal=sample_signal,
            account_balance=1000.0,
            open_positions=[],
            current_volatility=2.0,
        )
        assert isinstance(result, RiskAssessment)
        assert result.approved
    
    @pytest.mark.asyncio
    async def test_validate_trade_with_insufficient_balance(self, risk_controller, sample_signal):
        """Test trade validation with insufficient balance"""
        result = await risk_controller.validate_trade(
            signal=sample_signal,
            account_balance=50.0,
            open_positions=[],
            current_volatility=2.0,
        )
        assert isinstance(result, RiskAssessment)
        assert not result.approved
    
    @pytest.mark.asyncio
    async def test_max_open_positions_limit(self, risk_controller, sample_signal):
        """Test maximum open positions enforcement"""
        max_positions = risk_controller.max_open_positions
        open_positions = [{"symbol": f"cmt_symbol{i}"} for i in range(max_positions)]
        
        result = await risk_controller.validate_trade(
            signal=sample_signal,
            account_balance=1000.0,
            open_positions=open_positions,
            current_volatility=2.0,
        )
        assert not result.approved
        assert "maximum" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_volatility_filter(self, risk_controller, sample_signal):
        """Test volatility filter enforcement"""
        high_volatility = risk_controller.max_atr_percent + 1.0
        
        result = await risk_controller.validate_trade(
            signal=sample_signal,
            account_balance=1000.0,
            open_positions=[],
            current_volatility=high_volatility,
        )
        
        if risk_controller.volatility_filter_enabled:
            assert not result.approved
    
    def test_cooldown_mechanism(self, risk_controller):
        """Test trade cooldown mechanism"""
        symbol = "cmt_btcusdt"
        
        # No cooldown initially
        assert risk_controller._check_cooldown(symbol)
        
        # Record trade
        risk_controller.record_trade(symbol)
        
        # Cooldown should be active
        assert not risk_controller._check_cooldown(symbol)
    
    @pytest.mark.asyncio
    async def test_position_size_calculation(self, risk_controller, sample_signal):
        """Test position size calculation"""
        result = await risk_controller.validate_trade(
            signal=sample_signal,
            account_balance=1000.0,
            open_positions=[],
            current_volatility=2.0,
        )
        
        if result.approved:
            assert result.suggested_size > 0
            assert result.suggested_leverage >= risk_controller.min_leverage
            assert result.suggested_leverage <= risk_controller.max_leverage
    
    def test_leverage_limits(self, risk_controller):
        """Test leverage limit enforcement"""
        assert risk_controller.max_leverage <= 20  # WEEX competition limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
