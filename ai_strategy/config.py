"""
Configuration module for AI-Assisted Trading Strategy

Centralizes all configuration including:
- Binance API endpoints (public, no keys)
- WEEX API credentials (for live trading)
- Approved trading pairs
- Model parameters
- Feature engineering settings
- LLM provider configuration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class APIConfig:
    """API Configuration for data and execution"""
    
    # Binance (Public data - no authentication required)
    binance_base_url: str = "https://api.binance.com"
    binance_klines_endpoint: str = "/api/v3/klines"
    
    # WEEX (Live trading - requires authentication)
    # TODO: Fill these in when ready for live trading
    weex_base_url: str = "https://api.weex.com"
    weex_api_key: str = os.getenv("WEEX_API_KEY", "your_api_key_here")
    weex_secret: str = os.getenv("WEEX_SECRET", "your_secret_here")
    weex_passphrase: str = os.getenv("WEEX_PASSPHRASE", "your_passphrase_here")
    
    # Rate limiting
    max_requests_per_minute: int = 1200


@dataclass
class TradingPairsConfig:
    """Approved trading pairs for the competition"""
    
    # Only these 8 pairs are approved
    approved_pairs: List[str] = field(default_factory=lambda: [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "BNBUSDT",
        "LTCUSDT"
    ])
    
    # Timeframes to analyze
    timeframes: List[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    
    # Primary timeframe for trading decisions
    primary_timeframe: str = "1h"


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    
    # EMA periods
    ema_fast: int = 20
    ema_slow: int = 50
    ema_long: int = 200
    
    # ATR configuration
    atr_period: int = 14
    atr_multiplier: float = 1.5
    
    # RSI configuration
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # MACD configuration
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Return lookback periods
    return_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Volatility lookback
    volatility_period: int = 20


@dataclass
class ModelConfig:
    """Machine Learning model configuration"""
    
    # Model types to train
    models_to_train: List[str] = field(default_factory=lambda: [
        "random_forest",
        "xgboost",
        "logistic_regression"
    ])
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # Logistic Regression parameters
    lr_c: float = 1.0
    lr_max_iter: int = 1000
    
    # Training configuration
    train_test_split: float = 0.7  # 70% train, 30% test
    prediction_threshold: float = 0.52  # Minimum probability for signal (lowered for more trades)
    
    # Label generation
    label_forward_periods: int = 1  # Predict next period
    label_threshold: float = 0.003  # 0.3% minimum move to be labeled as up (lowered from 0.8%)
    
    # Feature selection
    use_feature_importance: bool = True
    min_feature_importance: float = 0.01


@dataclass
class LLMConfig:
    """LLM decision gate configuration"""
    
    # LLM Provider (OpenAI, Anthropic, Local, etc.)
    provider: str = "openai"  # or "anthropic", "local", "mock"
    
    # API configuration
    api_key: str = os.getenv("OPENAI_API_KEY", "your_openai_key_here")
    model: str = "gpt-4"  # or "gpt-3.5-turbo", "claude-3", etc.
    
    # Decision parameters
    use_llm_gate: bool = True  # Set to False to bypass LLM
    min_confidence: float = 0.7  # Minimum LLM confidence to approve
    max_retries: int = 3
    timeout_seconds: int = 10
    
    # Context window
    include_indicators: bool = True
    include_model_prediction: bool = True
    include_recent_trades: bool = True
    context_candles: int = 50  # Number of recent candles to include


@dataclass
class RiskConfig:
    """Risk management configuration"""
    
    # Position sizing
    initial_capital: float = 10000.0
    max_position_size_percent: float = 0.02  # 2% per trade
    max_leverage: float = 3.0
    
    # Stop loss / Take profit
    use_atr_stops: bool = True
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 3.0  # 2:1 risk/reward
    
    # Position limits
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    
    # Volatility filter
    min_atr_threshold: float = 0.001
    max_atr_threshold: float = 0.10
    
    # Cooldown
    cooldown_after_loss_minutes: int = 60
    cooldown_after_win_minutes: int = 30
    
    # Fees (Binance spot)
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    
    # Date range
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-26"
    
    # Data lookback for indicator calculation
    warmup_periods: int = 200
    
    # Execution
    simulate_execution_delay: bool = True
    execution_delay_seconds: int = 1
    
    # Logging
    log_all_signals: bool = True
    log_ml_predictions: bool = True
    log_llm_decisions: bool = True
    save_results_csv: bool = True


@dataclass
class AIStrategyConfig:
    """Master configuration class"""
    
    api: APIConfig = field(default_factory=APIConfig)
    pairs: TradingPairsConfig = field(default_factory=TradingPairsConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AIStrategyConfig':
        """Create config from dictionary"""
        return cls(
            api=APIConfig(**config_dict.get('api', {})),
            pairs=TradingPairsConfig(**config_dict.get('pairs', {})),
            features=FeatureConfig(**config_dict.get('features', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            llm=LLMConfig(**config_dict.get('llm', {})),
            risk=RiskConfig(**config_dict.get('risk', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {})),
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'api': self.api.__dict__,
            'pairs': self.pairs.__dict__,
            'features': self.features.__dict__,
            'model': self.model.__dict__,
            'llm': self.llm.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__,
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        # Check approved pairs
        if not self.pairs.approved_pairs:
            raise ValueError("No approved trading pairs configured")
        
        # Check model parameters
        if self.model.prediction_threshold < 0 or self.model.prediction_threshold > 1:
            raise ValueError("Model prediction threshold must be between 0 and 1")
        
        # Check risk parameters
        if self.risk.max_position_size_percent <= 0:
            raise ValueError("Max position size must be positive")
        
        if self.risk.max_leverage < 1:
            raise ValueError("Max leverage must be >= 1")
        
        return True


# Default configuration instance
default_config = AIStrategyConfig()


if __name__ == "__main__":
    # Example usage
    config = AIStrategyConfig()
    config.validate()
    
    print("=== AI Strategy Configuration ===")
    print(f"Approved pairs: {config.pairs.approved_pairs}")
    print(f"Primary timeframe: {config.pairs.primary_timeframe}")
    print(f"Models to train: {config.model.models_to_train}")
    print(f"Use LLM gate: {config.llm.use_llm_gate}")
    print(f"Initial capital: ${config.risk.initial_capital:,.2f}")
    print(f"Max leverage: {config.risk.max_leverage}x")
    print("\nConfiguration validated successfully!")
