"""
Machine Learning Model Engine for TrendCortex

This module provides comprehensive ML model training, prediction, and evaluation
capabilities for crypto trading strategy. It includes:

1. Feature Engineering: Creates trading features from OHLCV + indicators
2. Label Generation: Creates supervised learning labels from future returns
3. Model Training: Trains multiple ML models (RF, XGBoost, LogReg)
4. Prediction: Generates buy/sell predictions with probabilities
5. Evaluation: Comprehensive metrics and trading performance analysis
6. Model Persistence: Save/load trained models

Design Philosophy:
- Backtestable: Works entirely offline with historical data
- Time-series aware: Chronological train/test splits
- Feature-rich: Combines indicators, returns, and volatility
- Multi-model: Ensemble approach for robustness
- Explainable: Feature importance and prediction probabilities

Integration Points:
- Uses indicators from ../backtesting/indicators.py
- Integrates with LLM gate for final decision approval
- Feeds into backtester for strategy evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import joblib
import json
import logging

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

# Import XGBoost if available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: uv pip install xgboost")

# Import indicators from backtesting module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from backtesting.indicators import (
    compute_ema,
    compute_atr,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands
)

from ai_strategy.config import AIStrategyConfig, MODELS_DIR

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for trading ML models.
    
    Creates features from:
    - Technical indicators (EMA, ATR, RSI, MACD, BB)
    - Price returns (1, 3, 5, 10 periods)
    - Volatility measures
    - Relative strength metrics
    """
    
    def __init__(self, config: AIStrategyConfig):
        self.config = config
        self.feature_cols = []
    
    def create_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicator features to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = compute_ema(df['close'], self.config.features.ema_fast)
        df['ema_slow'] = compute_ema(df['close'], self.config.features.ema_slow)
        df['ema_long'] = compute_ema(df['close'], self.config.features.ema_long)
        
        # EMA relationships (key features!)
        df['ema_fast_slow_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_fast_slow_ratio'] = df['ema_fast'] / df['ema_slow']
        df['price_ema_fast_diff'] = df['close'] - df['ema_fast']
        df['price_ema_slow_diff'] = df['close'] - df['ema_slow']
        
        # ATR (volatility)
        df['atr'] = compute_atr(df, self.config.features.atr_period)
        df['atr_percent'] = df['atr'] / df['close']  # Normalized ATR
        
        # RSI
        df['rsi'] = compute_rsi(df['close'], self.config.features.rsi_period)
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Center around 0
        
        # MACD
        macd_line, signal_line, histogram = compute_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(
            df['close'],
            self.config.features.bb_period,
            self.config.features.bb_std
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        return df
    
    def create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create return-based features.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with added return columns
        """
        df = df.copy()
        
        # Returns over different periods
        for period in self.config.features.return_periods:
            df[f'return_{period}p'] = df['close'].pct_change(period)
        
        # Log returns
        df['log_return_1p'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            df: DataFrame with return data
        
        Returns:
            DataFrame with added volatility columns
        """
        df = df.copy()
        
        period = self.config.features.volatility_period
        
        # Rolling volatility (std of returns)
        df['volatility'] = df['return_1p'].rolling(period).std()
        
        # High-low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma'] = df['hl_range'].rolling(period).mean()
        
        # True range percentile
        df['tr_percentile'] = df['atr'].rolling(period).apply(
            lambda x: (x.iloc[-1] / x.mean()) if len(x) > 0 and x.mean() > 0 else 1.0
        )
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for ML model.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all features
        """
        df = self.create_indicator_features(df)
        df = self.create_return_features(df)
        df = self.create_volatility_features(df)
        
        # Store feature column names
        self.feature_cols = [
            'ema_fast_slow_diff', 'ema_fast_slow_ratio',
            'price_ema_fast_diff', 'price_ema_slow_diff',
            'atr_percent', 'rsi_normalized',
            'macd', 'macd_histogram',
            'bb_width', 'bb_position',
            'return_1p', 'return_3p', 'return_5p', 'return_10p',
            'log_return_1p', 'volume_change', 'volume_ma_ratio',
            'volatility', 'hl_range', 'tr_percentile'
        ]
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create supervised learning labels.
        
        Label Logic:
        - Look forward N periods (default: 1)
        - If future return > threshold: Label = 1 (UP)
        - If future return < -threshold: Label = 0 (DOWN)
        - Otherwise: Label = 0 (NEUTRAL - treated as DOWN)
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with 'label' and 'future_return' columns
        """
        df = df.copy()
        
        periods = self.config.model.label_forward_periods
        threshold = self.config.model.label_threshold
        
        # Calculate future return
        df['future_return'] = df['close'].shift(-periods) / df['close'] - 1
        
        # Create binary label: 1 for up, 0 for down/neutral
        df['label'] = (df['future_return'] > threshold).astype(int)
        
        # Alternative: multi-class labels (UP, DOWN, NEUTRAL)
        # df['label_multiclass'] = pd.cut(
        #     df['future_return'],
        #     bins=[-np.inf, -threshold, threshold, np.inf],
        #     labels=[0, 1, 2]  # DOWN, NEUTRAL, UP
        # )
        
        return df


class ModelEngine:
    """
    Main ML model engine for training and prediction.
    
    Supports multiple model types:
    - RandomForestClassifier (robust, interpretable)
    - XGBoost (high performance)
    - LogisticRegression (simple baseline)
    - MLPClassifier (neural network, optional)
    
    Features:
    - Chronological train/test splits (no shuffling)
    - Feature scaling for linear models
    - Model persistence (save/load)
    - Comprehensive evaluation metrics
    - Feature importance analysis
    """
    
    def __init__(self, config: AIStrategyConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_cols: List[str] = []
        self.training_metrics: Dict[str, Dict] = {}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        create_labels: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data with features and labels.
        
        Args:
            df: Raw OHLCV DataFrame
            create_labels: Whether to create labels (False for prediction only)
        
        Returns:
            DataFrame with features and optionally labels
        """
        # Create features
        df = self.feature_engineer.create_all_features(df)
        
        # Create labels if training
        if create_labels:
            df = self.feature_engineer.create_labels(df)
        
        # Store feature columns
        self.feature_cols = self.feature_engineer.feature_cols
        
        # Drop NaN rows (from indicators and returns)
        df = df.dropna()
        
        logger.info(f"Prepared data: {len(df)} rows, {len(self.feature_cols)} features")
        
        return df
    
    def train_test_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (time-series aware).
        
        IMPORTANT: No random shuffling! We preserve time order.
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            Tuple of (train_df, test_df)
        """
        split_point = int(len(df) * self.config.model.train_test_split)
        
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()
        
        logger.info(f"Train set: {len(train_df)} rows")
        logger.info(f"Test set: {len(test_df)} rows")
        
        return train_df, test_df
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> RandomForestClassifier:
        """Train Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=self.config.model.rf_n_estimators,
            max_depth=self.config.model.rf_max_depth,
            min_samples_split=self.config.model.rf_min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info("Trained Random Forest model")
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Optional[Any]:
        """Train XGBoost classifier"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping")
            return None
        
        model = XGBClassifier(
            n_estimators=self.config.model.xgb_n_estimators,
            max_depth=self.config.model.xgb_max_depth,
            learning_rate=self.config.model.xgb_learning_rate,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info("Trained XGBoost model")
        return model
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """Train Logistic Regression (with scaling)"""
        # Scale features for linear model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['logistic_regression'] = scaler
        
        model = LogisticRegression(
            C=self.config.model.lr_c,
            max_iter=self.config.model.lr_max_iter,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        logger.info("Trained Logistic Regression model")
        return model
    
    def train_models(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            df: DataFrame with features and labels
        
        Returns:
            Dictionary of trained models
        """
        # Prepare data
        df = self.prepare_data(df, create_labels=True)
        
        # Split data
        train_df, test_df = self.train_test_split(df)
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['label']
        X_test = test_df[self.feature_cols]
        y_test = test_df['label']
        
        # Train each model
        if "random_forest" in self.config.model.models_to_train:
            self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        
        if "xgboost" in self.config.model.models_to_train:
            xgb_model = self.train_xgboost(X_train, y_train)
            if xgb_model:
                self.models['xgboost'] = xgb_model
        
        if "logistic_regression" in self.config.model.models_to_train:
            self.models['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        
        # Evaluate all models
        for name, model in self.models.items():
            metrics = self.evaluate_model(name, model, X_test, y_test)
            self.training_metrics[name] = metrics
        
        logger.info(f"Trained {len(self.models)} models successfully")
        
        return self.models
    
    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate model performance.
        
        Returns classification metrics and trading-specific metrics.
        """
        # Scale if needed
        if model_name in self.scalers:
            X_test_scaled = self.scalers[model_name].transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0
        
        # Trading metrics
        # Apply threshold and calculate returns
        threshold = self.config.model.prediction_threshold
        trades = y_proba >= threshold
        
        if trades.sum() > 0:
            # Simplified directional accuracy
            trade_accuracy = accuracy_score(y_test[trades], y_pred[trades])
        else:
            trade_accuracy = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'trade_accuracy': trade_accuracy,
            'total_predictions': len(y_test),
            'positive_predictions': y_pred.sum(),
            'trades_above_threshold': trades.sum()
        }
        
        logger.info(f"\n{model_name} Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Trade Accuracy: {trade_accuracy:.4f}")
        logger.info(f"  Trades (>{threshold}): {trades.sum()}/{len(y_test)}")
        
        return metrics
    
    def predict_signals(
        self,
        df: pd.DataFrame,
        model_name: str = "random_forest"
    ) -> pd.DataFrame:
        """
        Generate predictions for new data.
        
        Args:
            df: DataFrame with OHLCV data
            model_name: Name of model to use
        
        Returns:
            DataFrame with predictions and probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Prepare features
        df = self.prepare_data(df, create_labels=False)
        X = df[self.feature_cols]
        
        # Get model
        model = self.models[model_name]
        
        # Scale if needed
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
        
        # Add to dataframe
        df['ml_prediction'] = predictions
        df['ml_probability'] = probabilities
        df['ml_signal'] = (probabilities >= self.config.model.prediction_threshold).astype(int)
        
        return df
    
    def get_feature_importance(
        self,
        model_name: str = "random_forest",
        top_n: int = 10
    ) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
    
    def save_models(self, suffix: str = ""):
        """Save all trained models to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            filename = f"model_{name}_{timestamp}{suffix}.joblib"
            filepath = MODELS_DIR / filename
            joblib.dump(model, filepath)
            logger.info(f"Saved {name} model to {filepath}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            filename = f"scaler_{name}_{timestamp}{suffix}.joblib"
            filepath = MODELS_DIR / filename
            joblib.dump(scaler, filepath)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'feature_cols': self.feature_cols,
            'models': list(self.models.keys()),
            'metrics': self.training_metrics,
            'config': self.config.model.__dict__
        }
        
        metadata_file = MODELS_DIR / f"metadata_{timestamp}{suffix}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved model metadata to {metadata_file}")
    
    def load_models(self, timestamp: str, suffix: str = ""):
        """Load previously trained models"""
        for name in self.config.model.models_to_train:
            filename = f"model_{name}_{timestamp}{suffix}.joblib"
            filepath = MODELS_DIR / filename
            
            if filepath.exists():
                self.models[name] = joblib.load(filepath)
                logger.info(f"Loaded {name} model from {filepath}")
            
            # Load scaler if exists
            scaler_file = MODELS_DIR / f"scaler_{name}_{timestamp}{suffix}.joblib"
            if scaler_file.exists():
                self.scalers[name] = joblib.load(scaler_file)
        
        # Load metadata
        metadata_file = MODELS_DIR / f"metadata_{timestamp}{suffix}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.feature_cols = metadata['feature_cols']
                self.training_metrics = metadata.get('metrics', {})


if __name__ == "__main__":
    # Example usage
    from ai_strategy.config import AIStrategyConfig
    import sys
    import os
    import importlib.util
    
    # Add parent and backtesting to path
    parent_dir = str(Path(__file__).parent.parent)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'backtesting'))
    
    # Import data fetcher (now config will be found from root)
    spec = importlib.util.spec_from_file_location(
        "data_fetcher_module",
        os.path.join(parent_dir, "backtesting", "data_fetcher.py")
    )
    if spec and spec.loader:
        data_fetcher_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_fetcher_module)
        get_historical_data = data_fetcher_module.get_historical_data
    else:
        print("Error: Could not load data_fetcher module")
        print("Run this script from project root directory")
        sys.exit(1)
    
    from datetime import datetime, timedelta
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config
    config = AIStrategyConfig()
    
    # Fetch data
    print("Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    df = get_historical_data("BTCUSDT", "1h", start_date, end_date)
    
    # Create model engine
    print("\nTraining models...")
    engine = ModelEngine(config)
    
    # Train models
    models = engine.train_models(df)
    
    # Feature importance
    print("\nTop 10 Features (Random Forest):")
    print(engine.get_feature_importance())
    
    # Save models
    engine.save_models(suffix="_demo")
    
    # Generate predictions on recent data
    print("\nGenerating predictions on recent data...")
    recent_df = df.tail(100).copy()
    predictions = engine.predict_signals(recent_df, model_name="random_forest")
    
    print(f"\nSignals generated: {predictions['ml_signal'].sum()}")
    print(f"Average probability: {predictions['ml_probability'].mean():.4f}")
    
    print("\n✅ Model training complete!")
