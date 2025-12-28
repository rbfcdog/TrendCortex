"""
Configuration Management Module

Handles loading and validation of configuration from JSON files and environment variables.
Provides type-safe access to all system settings.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class APIConfig(BaseModel):
    """WEEX API configuration"""
    base_url: str = "https://api-contract.weex.com"
    key: str
    secret: str
    passphrase: str
    rate_limit_delay: float = 0.1
    max_retries: int = 3
    timeout: int = 30
    locale: str = "en-US"


class TradingConfig(BaseModel):
    """Trading parameters configuration"""
    symbols: List[str]
    primary_symbol: str = "cmt_btcusdt"
    timeframes: Dict[str, str]
    default_leverage: int = 5
    max_leverage: int = 20
    margin_mode: int = 1
    order_type: Dict[str, str]
    position_side: Dict[str, str]


class RiskConfig(BaseModel):
    """Risk management configuration"""
    max_position_size_usdt: float = 500.0
    max_position_size_percent: float = 10.0
    max_leverage: int = 20
    min_leverage: int = 1
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 5.0
    trailing_stop_percent: float = 1.5
    max_portfolio_risk_percent: float = 5.0
    max_daily_loss_percent: float = 10.0
    cooldown_seconds: int = 300
    min_account_balance_usdt: float = 100.0
    volatility_filter: Dict[str, Any]
    correlation_limit: Dict[str, Any]


class IndicatorsConfig(BaseModel):
    """Technical indicators configuration"""
    ema: Dict[str, int]
    rsi: Dict[str, Any]
    atr: Dict[str, Any]
    bollinger_bands: Dict[str, Any]
    macd: Dict[str, int]
    volume: Dict[str, Any]


class SignalsConfig(BaseModel):
    """Signal generation configuration"""
    enable_long: bool = True
    enable_short: bool = True
    min_confidence: float = 0.6
    require_trend_alignment: bool = True
    require_volume_confirmation: bool = False
    signal_timeout_seconds: int = 60
    filters: Dict[str, float]


class MLModelConfig(BaseModel):
    """Machine learning model configuration"""
    enabled: bool = True
    model_path: str = "models/trained_model.pkl"
    min_confidence: float = 0.65
    feature_window: int = 100
    retrain_interval_hours: int = 24
    use_online_learning: bool = False
    features: List[str]


class LLMConfig(BaseModel):
    """LLM decision gate configuration"""
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    min_confidence: float = 0.7
    timeout: int = 10
    use_as_veto: bool = True
    require_explanation: bool = True
    prompt_template: str


class ExecutionConfig(BaseModel):
    """Order execution configuration"""
    dry_run: bool = True
    enable_paper_trading: bool = False
    order_timeout_seconds: int = 30
    fill_timeout_seconds: int = 60
    partial_fill_handling: str = "accept"
    slippage_tolerance_percent: float = 0.5
    use_limit_orders: bool = True
    limit_order_offset_percent: float = 0.1
    cancel_unfilled_after_seconds: int = 120


class DataConfig(BaseModel):
    """Data management configuration"""
    cache_enabled: bool = True
    cache_directory: str = "data/cache"
    cache_ttl_seconds: int = 60
    historical_data_path: str = "data/historical"
    max_candles_fetch: int = 1000
    warmup_periods: int = 200


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    json_format: bool = True
    log_directory: str = "logs"
    rotate_logs: bool = True
    max_log_size_mb: int = 100
    backup_count: int = 10
    log_components: Dict[str, bool]


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    enabled: bool = True
    metrics_interval_seconds: int = 60
    alert_on_error: bool = False
    alert_on_large_drawdown: bool = True
    drawdown_threshold_percent: float = 15.0
    email_alerts: bool = False
    telegram_alerts: bool = False


class SystemConfig(BaseModel):
    """System-level configuration"""
    loop_interval_seconds: int = 5
    max_concurrent_tasks: int = 10
    graceful_shutdown_timeout: int = 30
    health_check_interval: int = 60
    auto_restart_on_error: bool = False


class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    enabled: bool = False
    start_date: str = "2025-01-01"
    end_date: str = "2025-01-31"
    initial_balance: float = 1000.0
    commission_rate: float = 0.0008
    slippage_model: str = "fixed"


class Config:
    """
    Main configuration class that loads and provides access to all settings.
    
    Usage:
        config = Config.load("config.json")
        api_key = config.api.key
        symbols = config.trading.symbols
    """
    
    def __init__(
        self,
        api: APIConfig,
        trading: TradingConfig,
        risk: RiskConfig,
        indicators: IndicatorsConfig,
        signals: SignalsConfig,
        ml_model: MLModelConfig,
        llm: LLMConfig,
        execution: ExecutionConfig,
        data: DataConfig,
        logging: LoggingConfig,
        monitoring: MonitoringConfig,
        system: SystemConfig,
        backtest: BacktestConfig,
    ):
        self.api = api
        self.trading = trading
        self.risk = risk
        self.indicators = indicators
        self.signals = signals
        self.ml_model = ml_model
        self.llm = llm
        self.execution = execution
        self.data = data
        self.logging = logging
        self.monitoring = monitoring
        self.system = system
        self.backtest = backtest
    
    @classmethod
    def load(cls, config_path: str = "config.json") -> "Config":
        """
        Load configuration from JSON file with environment variable overrides.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Config instance with validated settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please copy config.example.json to config.json and configure your settings."
            )
        
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        # Override with environment variables if present
        config_data = cls._apply_env_overrides(config_data)
        
        # Validate and create typed config objects
        return cls(
            api=APIConfig(**config_data["api"]),
            trading=TradingConfig(**config_data["trading"]),
            risk=RiskConfig(**config_data["risk"]),
            indicators=IndicatorsConfig(**config_data["indicators"]),
            signals=SignalsConfig(**config_data["signals"]),
            ml_model=MLModelConfig(**config_data["ml_model"]),
            llm=LLMConfig(**config_data["llm"]),
            execution=ExecutionConfig(**config_data["execution"]),
            data=DataConfig(**config_data["data"]),
            logging=LoggingConfig(**config_data["logging"]),
            monitoring=MonitoringConfig(**config_data["monitoring"]),
            system=SystemConfig(**config_data["system"]),
            backtest=BacktestConfig(**config_data["backtest"]),
        )
    
    @staticmethod
    def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables should be prefixed with TRENDCORTEX_ and use
        double underscores for nested keys, e.g.:
        TRENDCORTEX_API__KEY=abc123
        TRENDCORTEX_TRADING__MAX_LEVERAGE=20
        """
        prefix = "TRENDCORTEX_"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # Parse nested keys
            key_path = env_key[len(prefix):].lower().split("__")
            
            # Navigate to nested dict
            current = config_data
            for key in key_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set value with type conversion
            final_key = key_path[-1]
            current[final_key] = cls._convert_env_value(env_value)
        
        return config_data
    
    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON (for lists/dicts)
        if value.startswith(("[", "{")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String
        return value
    
    def validate_trading_rules(self) -> None:
        """
        Validate that configuration complies with WEEX competition rules.
        
        Raises:
            ValueError: If configuration violates competition rules
        """
        # Check max leverage
        if self.risk.max_leverage > 20:
            raise ValueError(
                f"Maximum leverage {self.risk.max_leverage}x exceeds competition limit of 20x"
            )
        
        if self.trading.max_leverage > 20:
            raise ValueError(
                f"Trading max leverage {self.trading.max_leverage}x exceeds competition limit of 20x"
            )
        
        # Check allowed symbols
        allowed_symbols = {
            "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", "cmt_dogeusdt",
            "cmt_xrpusdt", "cmt_adausdt", "cmt_bnbusdt", "cmt_ltcusdt"
        }
        
        for symbol in self.trading.symbols:
            if symbol not in allowed_symbols:
                raise ValueError(
                    f"Symbol {symbol} is not allowed in WEEX competition. "
                    f"Allowed symbols: {allowed_symbols}"
                )
        
        # Check API credentials are set
        if self.api.key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "API credentials not configured. Please update config.json with your WEEX API credentials."
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api": self.api.dict(),
            "trading": self.trading.dict(),
            "risk": self.risk.dict(),
            "indicators": self.indicators.dict(),
            "signals": self.signals.dict(),
            "ml_model": self.ml_model.dict(),
            "llm": self.llm.dict(),
            "execution": self.execution.dict(),
            "data": self.data.dict(),
            "logging": self.logging.dict(),
            "monitoring": self.monitoring.dict(),
            "system": self.system.dict(),
            "backtest": self.backtest.dict(),
        }
