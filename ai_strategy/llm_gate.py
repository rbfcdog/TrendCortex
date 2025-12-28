"""
LLM Decision Gate for TrendCortex AI Strategy

This module provides an LLM-powered decision gate that evaluates trading candidates
before execution. It interprets technical indicators, ML predictions, and market
context to provide:

1. Approve/Reject Decision: Binary go/no-go for each trade
2. Confidence Score: How confident the LLM is (0-1)
3. Reasoning: Human-readable explanation of the decision

Design Philosophy:
- Explainable AI: Every decision comes with reasoning
- Context-aware: Considers recent market conditions
- Auditable: All decisions logged for competition compliance
- Flexible: Supports multiple LLM providers (OpenAI, Anthropic, local)
- Graceful degradation: Can bypass LLM if unavailable

Integration:
- Receives ML predictions from model_engine.py
- Receives indicators from indicators.py
- Feeds into execution layer with approval + reasoning
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: uv pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ai_strategy.config import AIStrategyConfig

logger = logging.getLogger(__name__)


class LLMGate:
    """
    LLM-powered decision gate for trading signals.
    
    Evaluates each trading candidate by:
    1. Gathering context (indicators, ML prediction, recent trades)
    2. Formatting prompt for LLM
    3. Calling LLM API
    4. Parsing decision with confidence and reasoning
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5-turbo)
    - Anthropic (Claude)
    - Local models (via API)
    - Mock mode (for testing without API)
    """
    
    def __init__(self, config: AIStrategyConfig):
        self.config = config
        self.llm_config = config.llm
        
        # Initialize LLM client based on provider
        self.client = None
        if self.llm_config.use_llm_gate:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on configuration"""
        provider = self.llm_config.provider.lower()
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI library not installed")
                return
            
            if self.llm_config.api_key == "your_openai_key_here":
                logger.warning("OpenAI API key not configured. Using mock mode.")
                return
            
            self.client = openai.OpenAI(api_key=self.llm_config.api_key)
            logger.info("Initialized OpenAI client")
        
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                logger.error("Anthropic library not installed")
                return
            
            if self.llm_config.api_key == "your_anthropic_key_here":
                logger.warning("Anthropic API key not configured. Using mock mode.")
                return
            
            self.client = anthropic.Anthropic(api_key=self.llm_config.api_key)
            logger.info("Initialized Anthropic client")
        
        elif provider == "mock":
            logger.info("Using mock LLM mode (no API calls)")
            self.client = None
        
        else:
            logger.warning(f"Unknown provider {provider}. Using mock mode.")
    
    def format_context(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict[str, float],
        ml_prediction: Dict[str, Any],
        recent_candles: Optional[List[Dict]] = None,
        recent_trades: Optional[List[Dict]] = None
    ) -> str:
        """
        Format trading context into a prompt for LLM.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            current_price: Current price
            indicators: Dict of indicator values (EMA, ATR, RSI, etc.)
            ml_prediction: ML model output (prediction, probability)
            recent_candles: Recent OHLCV data
            recent_trades: Recent trade history
        
        Returns:
            Formatted context string
        """
        context = f"""
# Trading Signal Evaluation Request

## Symbol
{symbol}

## Current Market State
- Price: ${current_price:.2f}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Technical Indicators
"""
        
        # Add indicators
        if self.llm_config.include_indicators:
            for key, value in indicators.items():
                if isinstance(value, float):
                    context += f"- {key}: {value:.4f}\n"
                else:
                    context += f"- {key}: {value}\n"
        
        # Add ML prediction
        if self.llm_config.include_model_prediction:
            context += f"""
## Machine Learning Prediction
- Prediction: {"BUY" if ml_prediction.get('prediction', 0) == 1 else "HOLD"}
- Confidence: {ml_prediction.get('probability', 0):.2%}
- Model: {ml_prediction.get('model', 'unknown')}
"""
        
        # Add recent candles
        if self.llm_config.include_indicators and recent_candles:
            context += f"\n## Recent Price Action (last {len(recent_candles)} candles)\n"
            for i, candle in enumerate(recent_candles[-5:]):  # Last 5 candles
                context += f"- Candle {i+1}: O:{candle.get('open', 0):.2f} "
                context += f"H:{candle.get('high', 0):.2f} "
                context += f"L:{candle.get('low', 0):.2f} "
                context += f"C:{candle.get('close', 0):.2f}\n"
        
        # Add recent trades
        if self.llm_config.include_recent_trades and recent_trades:
            context += f"\n## Recent Trades (last {len(recent_trades)})\n"
            for trade in recent_trades[-3:]:  # Last 3 trades
                context += f"- {trade.get('side', 'N/A')}: "
                context += f"Entry ${trade.get('entry_price', 0):.2f} → "
                context += f"Exit ${trade.get('exit_price', 0):.2f}, "
                context += f"P&L: {trade.get('pnl_percent', 0):.2f}%\n"
        
        return context
    
    def format_prompt(self, context: str) -> str:
        """
        Create the full LLM prompt with instructions.
        
        Args:
            context: Trading context from format_context()
        
        Returns:
            Complete prompt for LLM
        """
        prompt = f"""You are an expert cryptocurrency trading advisor. Your task is to evaluate a trading signal and decide whether to approve or reject it.

{context}

## Your Task
Based on the information above, decide whether to approve this trading signal.

Consider:
1. **Technical indicators**: Do they support the signal? Are there any concerning divergences?
2. **ML prediction confidence**: Is the model sufficiently confident?
3. **Risk factors**: Are there any red flags (extreme volatility, overbought/oversold conditions)?
4. **Market context**: Does the recent price action support this signal?

## Response Format
Respond with a JSON object containing:
- "approve": true or false
- "confidence": a number between 0 and 1 representing your confidence
- "reasoning": a brief explanation (2-3 sentences) of your decision

Example:
{{
  "approve": true,
  "confidence": 0.85,
  "reasoning": "The ML model shows high confidence (75%) aligned with bullish EMA crossover. RSI at 55 indicates room for upside without overbought concerns. Recent price action confirms upward momentum."
}}

Your response (JSON only, no additional text):"""
        
        return prompt
    
    def call_openai(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API.
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            LLM response dict
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": "You are a professional trading advisor. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent decisions
                max_tokens=500,
                timeout=self.llm_config.timeout_seconds
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            result = json.loads(content)
            
            return {
                "approve": result.get("approve", False),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
                "raw_response": content
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {content}")
            return {
                "approve": False,
                "confidence": 0.0,
                "reasoning": f"Error parsing LLM response: {e}",
                "raw_response": content
            }
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "approve": False,
                "confidence": 0.0,
                "reasoning": f"LLM API error: {str(e)}",
                "error": str(e)
            }
    
    def call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """
        Call Anthropic Claude API.
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            LLM response dict
        """
        try:
            response = self.client.messages.create(
                model=self.llm_config.model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=self.llm_config.timeout_seconds
            )
            
            content = response.content[0].text
            
            # Parse JSON response
            result = json.loads(content)
            
            return {
                "approve": result.get("approve", False),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
                "raw_response": content
            }
        
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {
                "approve": False,
                "confidence": 0.0,
                "reasoning": f"LLM API error: {str(e)}",
                "error": str(e)
            }
    
    def mock_decision(
        self,
        ml_prediction: Dict[str, Any],
        indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Mock LLM decision for testing without API.
        
        Uses simple rules to simulate LLM reasoning:
        - Approve if ML probability > 0.6
        - Approve if EMA fast > EMA slow and RSI < 70
        - Higher confidence if both conditions align
        
        Args:
            ml_prediction: ML model output
            indicators: Technical indicators
        
        Returns:
            Simulated LLM response
        """
        ml_prob = ml_prediction.get('probability', 0.0)
        ml_pred = ml_prediction.get('prediction', 0)
        
        ema_bullish = indicators.get('ema_fast', 0) > indicators.get('ema_slow', 0)
        rsi = indicators.get('rsi', 50)
        rsi_ok = 30 < rsi < 70
        
        # Simple rule-based decision
        approve = False
        confidence = 0.5
        reasoning = ""
        
        if ml_pred == 1 and ml_prob > 0.6:
            if ema_bullish and rsi_ok:
                approve = True
                confidence = min(0.85, ml_prob + 0.15)
                reasoning = f"Strong ML signal ({ml_prob:.0%}) aligned with bullish EMA crossover. RSI at {rsi:.0f} indicates healthy momentum."
            elif ema_bullish:
                approve = True
                confidence = ml_prob
                reasoning = f"ML prediction ({ml_prob:.0%}) supported by EMA trend, though RSI at {rsi:.0f} warrants caution."
            else:
                approve = False
                confidence = 0.3
                reasoning = f"ML shows {ml_prob:.0%} confidence but EMA trend not aligned. Waiting for better setup."
        else:
            approve = False
            confidence = 0.2
            reasoning = f"ML confidence too low ({ml_prob:.0%}) for entry. Indicators don't provide strong confirmation."
        
        return {
            "approve": approve,
            "confidence": confidence,
            "reasoning": reasoning,
            "mock": True
        }
    
    def evaluate_candidate(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict[str, float],
        ml_prediction: Dict[str, Any],
        recent_candles: Optional[List[Dict]] = None,
        recent_trades: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Main evaluation method. Evaluates a trading candidate using LLM.
        
        Args:
            symbol: Trading pair
            current_price: Current price
            indicators: Technical indicators
            ml_prediction: ML model prediction
            recent_candles: Recent price data
            recent_trades: Recent trade history
        
        Returns:
            Dict containing:
            - approve_trade: bool
            - confidence: float (0-1)
            - explanation: str
            - timestamp: str
            - provider: str
        """
        start_time = time.time()
        
        # If LLM gate is disabled, use simple threshold
        if not self.llm_config.use_llm_gate:
            ml_prob = ml_prediction.get('probability', 0.0)
            return {
                "approve_trade": ml_prob >= self.config.model.prediction_threshold,
                "confidence": ml_prob,
                "explanation": f"ML threshold decision (no LLM): {ml_prob:.2%}",
                "timestamp": datetime.now().isoformat(),
                "provider": "threshold_only",
                "execution_time": time.time() - start_time
            }
        
        # Format context
        context = self.format_context(
            symbol=symbol,
            current_price=current_price,
            indicators=indicators,
            ml_prediction=ml_prediction,
            recent_candles=recent_candles,
            recent_trades=recent_trades
        )
        
        # Format prompt
        prompt = self.format_prompt(context)
        
        # Call LLM (with retries)
        result = None
        for attempt in range(self.llm_config.max_retries):
            try:
                if self.llm_config.provider == "openai" and self.client:
                    result = self.call_openai(prompt)
                elif self.llm_config.provider == "anthropic" and self.client:
                    result = self.call_anthropic(prompt)
                else:
                    # Mock mode
                    result = self.mock_decision(ml_prediction, indicators)
                
                if result:
                    break
            
            except Exception as e:
                logger.error(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.llm_config.max_retries - 1:
                    # Final attempt failed, use mock
                    result = self.mock_decision(ml_prediction, indicators)
        
        # Format response
        execution_time = time.time() - start_time
        
        return {
            "approve_trade": result.get("approve", False) and result.get("confidence", 0.0) >= self.llm_config.min_confidence,
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("reasoning", ""),
            "timestamp": datetime.now().isoformat(),
            "provider": self.llm_config.provider if not result.get("mock") else "mock",
            "execution_time": execution_time,
            "raw_llm_response": result
        }


if __name__ == "__main__":
    # Example usage
    from ai_strategy.config import AIStrategyConfig
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config
    config = AIStrategyConfig()
    config.llm.provider = "mock"  # Use mock mode for testing
    config.llm.use_llm_gate = True
    
    # Create LLM gate
    llm_gate = LLMGate(config)
    
    # Example candidate
    indicators = {
        'ema_fast': 50000.0,
        'ema_slow': 49000.0,
        'ema_long': 48000.0,
        'rsi': 55.0,
        'atr': 500.0,
        'macd': 150.0
    }
    
    ml_prediction = {
        'prediction': 1,
        'probability': 0.75,
        'model': 'random_forest'
    }
    
    # Evaluate
    print("\n=== LLM Gate Evaluation ===")
    decision = llm_gate.evaluate_candidate(
        symbol="BTCUSDT",
        current_price=50000.0,
        indicators=indicators,
        ml_prediction=ml_prediction
    )
    
    print(f"\nDecision: {'APPROVE' if decision['approve_trade'] else 'REJECT'}")
    print(f"Confidence: {decision['confidence']:.2%}")
    print(f"Reasoning: {decision['explanation']}")
    print(f"Provider: {decision['provider']}")
    print(f"Execution time: {decision['execution_time']:.3f}s")
    
    print("\n✅ LLM Gate test complete!")
