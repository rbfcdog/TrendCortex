"""
LIVE IMPLEMENTATION - Quick Start Guide
Deploy the profitable deterministic strategies
"""

from optimized_strategies import VolumeProfileBreakout, ConservativeBreakout
from data_fetcher import BinanceDataFetcher
import pandas as pd
from datetime import datetime


class LiveTrader:
    """
    Simple live trading implementation
    Runs both winning strategies in parallel
    """
    
    def __init__(self, capital: float = 10000):
        self.capital = capital
        self.risk_per_trade = 0.01  # 1%
        self.stop_loss_pct = 0.02   # 2%
        self.take_profit_pct = 0.04 # 4%
        
        # Initialize strategies
        self.strategy_1h = VolumeProfileBreakout()
        self.strategy_4h = ConservativeBreakout()
        
        # Positions
        self.position_1h = None
        self.position_4h = None
        
    def calculate_position_size(self, entry_price: float) -> float:
        """Calculate position size based on 1% risk"""
        risk_amount = self.capital * self.risk_per_trade
        stop_distance = entry_price * self.stop_loss_pct
        position_size = risk_amount / stop_distance
        
        # Don't use more than 10% of capital
        max_position = self.capital * 0.10
        if position_size * entry_price > max_position:
            position_size = max_position / entry_price
            
        return position_size
    
    def check_signals_1h(self):
        """Check 1H signals for Volume Profile Breakout"""
        fetcher = BinanceDataFetcher()
        df = fetcher.get_historical_klines("BTCUSDT", "1h", 7)
        
        if df is None:
            return None
            
        df = self.strategy_1h.generate_signals(df)
        
        # Check last candle for signal
        if df.iloc[-1]['signal'] == 1:
            return {
                'timeframe': '1h',
                'strategy': 'Volume Profile Breakout',
                'entry_price': df.iloc[-1]['close'],
                'stop_loss': df.iloc[-1]['close'] * (1 - self.stop_loss_pct),
                'take_profit': df.iloc[-1]['close'] * (1 + self.take_profit_pct),
                'timestamp': datetime.now()
            }
        
        return None
    
    def check_signals_4h(self):
        """Check 4H signals for Conservative Breakout"""
        fetcher = BinanceDataFetcher()
        df = fetcher.get_historical_klines("BTCUSDT", "4h", 30)
        
        if df is None:
            return None
            
        df = self.strategy_4h.generate_signals(df)
        
        # Check last candle for signal
        if df.iloc[-1]['signal'] == 1:
            return {
                'timeframe': '4h',
                'strategy': 'Conservative Breakout',
                'entry_price': df.iloc[-1]['close'],
                'stop_loss': df.iloc[-1]['close'] * (1 - self.stop_loss_pct),
                'take_profit': df.iloc[-1]['close'] * (1 + self.take_profit_pct),
                'timestamp': datetime.now()
            }
        
        return None
    
    def run(self):
        """
        Main loop - check for signals
        In production, run this every hour
        """
        print("=" * 80)
        print("🤖 LIVE TRADING BOT")
        print("=" * 80)
        print(f"\nCapital: ${self.capital:,.2f}")
        print(f"Risk per trade: {self.risk_per_trade * 100}%")
        print(f"Stop Loss: {self.stop_loss_pct * 100}%")
        print(f"Take Profit: {self.take_profit_pct * 100}%")
        print("\nStrategies:")
        print("  1. Volume Profile Breakout (1H)")
        print("  2. Conservative Breakout (4H)")
        print("\n" + "=" * 80)
        
        # Check 1H signals
        print("\n🔍 Checking 1H signals...")
        signal_1h = self.check_signals_1h()
        
        if signal_1h:
            position_size = self.calculate_position_size(signal_1h['entry_price'])
            print("\n🚨 SIGNAL DETECTED (1H)!")
            print(f"   Strategy: {signal_1h['strategy']}")
            print(f"   Entry Price: ${signal_1h['entry_price']:,.2f}")
            print(f"   Stop Loss: ${signal_1h['stop_loss']:,.2f}")
            print(f"   Take Profit: ${signal_1h['take_profit']:,.2f}")
            print(f"   Position Size: {position_size:.6f} BTC")
            print(f"   Capital at Risk: ${self.capital * self.risk_per_trade:,.2f}")
            print(f"   Potential Profit: ${position_size * signal_1h['entry_price'] * self.take_profit_pct:,.2f}")
        else:
            print("   ✅ No signals (waiting for setup)")
        
        # Check 4H signals
        print("\n🔍 Checking 4H signals...")
        signal_4h = self.check_signals_4h()
        
        if signal_4h:
            position_size = self.calculate_position_size(signal_4h['entry_price'])
            print("\n🚨 SIGNAL DETECTED (4H)!")
            print(f"   Strategy: {signal_4h['strategy']}")
            print(f"   Entry Price: ${signal_4h['entry_price']:,.2f}")
            print(f"   Stop Loss: ${signal_4h['stop_loss']:,.2f}")
            print(f"   Take Profit: ${signal_4h['take_profit']:,.2f}")
            print(f"   Position Size: {position_size:.6f} BTC")
            print(f"   Capital at Risk: ${self.capital * self.risk_per_trade:,.2f}")
            print(f"   Potential Profit: ${position_size * signal_4h['entry_price'] * self.take_profit_pct:,.2f}")
        else:
            print("   ✅ No signals (waiting for setup)")
        
        print("\n" + "=" * 80)
        print("✅ Scan complete")
        print("=" * 80)
        
        return {
            'signal_1h': signal_1h,
            'signal_4h': signal_4h
        }


def main():
    """
    Quick start - run the bot once
    
    For continuous operation:
    1. Set up a cron job to run every hour
    2. Or use a scheduler (APScheduler)
    3. Or run in a while loop with sleep
    """
    trader = LiveTrader(capital=10000)
    signals = trader.run()
    
    if signals['signal_1h'] or signals['signal_4h']:
        print("\n💡 ACTION REQUIRED:")
        print("   1. Review the signal details above")
        print("   2. Verify current price on Binance")
        print("   3. Place limit order at entry price")
        print("   4. Set stop loss and take profit orders")
        print("   5. Monitor position until close")
    else:
        print("\n💤 No action needed - keep monitoring")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                   🤖 DETERMINISTIC TRADING BOT                             ║
    ║                                                                            ║
    ║  Strategies: Volume Profile Breakout (1H) + Conservative Breakout (4H)    ║
    ║  Risk: 1% per trade | R/R: 2:1 | Capital Preservation: Strict             ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    ⚠️  DISCLAIMER:
    This is for PAPER TRADING / TESTING only!
    Do NOT use real money until you've verified results in paper trading.
    
    Expected Performance:
    • Win Rate: 40-50%
    • Annual Return: 1.5-2.0%
    • Max Drawdown: <1%
    • Trades: ~2-4 per month
    
    """)
    
    main()
