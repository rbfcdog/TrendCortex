"""
Production Bot Runner with Health Monitoring and Auto-Restart

This module provides a production-ready framework for running TrendCortex 24/7
with health monitoring, automatic restarts, and graceful error handling.
"""

import asyncio
import signal
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import psutil
import os

from trendcortex.config import Config
from trendcortex.logger import get_logger
from health_server import HealthCheckServer


logger = get_logger()


class HealthMonitor:
    """Monitors bot health and system resources"""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.error_count = 0
        self.restart_count = 0
        self.max_errors_per_hour = 10
        self.max_restarts_per_day = 20
        
    def update_heartbeat(self):
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()
        
    def record_error(self):
        """Record an error occurrence"""
        self.error_count += 1
        
    def record_restart(self):
        """Record a restart"""
        self.restart_count += 1
        
    def is_healthy(self) -> bool:
        """Quick health check"""
        is_healthy, _ = self.check_health()
        return is_healthy
        
    def check_health(self) -> tuple[bool, str]:
        """
        Check if the bot is healthy.
        
        Returns:
            Tuple of (is_healthy, reason)
        """
        # Check heartbeat (should update every 60 seconds)
        time_since_heartbeat = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        if time_since_heartbeat > 300:  # 5 minutes
            return False, f"No heartbeat for {time_since_heartbeat:.0f} seconds"
        
        # Check error rate
        uptime_hours = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        if uptime_hours > 0:
            error_rate = self.error_count / uptime_hours
            if error_rate > self.max_errors_per_hour:
                return False, f"Error rate too high: {error_rate:.1f}/hour"
        
        # Check restart count
        uptime_days = (datetime.utcnow() - self.start_time).total_seconds() / 86400
        if uptime_days > 0:
            restart_rate = self.restart_count / uptime_days
            if restart_rate > self.max_restarts_per_day:
                return False, f"Restart rate too high: {restart_rate:.1f}/day"
        
        # Check system resources
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            return False, f"Memory usage too high: {memory_percent:.1f}%"
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return False, f"CPU usage too high: {cpu_percent:.1f}%"
        
        return True, "Healthy"
    
    def get_metrics(self) -> dict:
        """Get current health metrics for health server"""
        return {
            "restart_count": self.restart_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, (datetime.utcnow() - self.start_time).total_seconds() / 3600),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "last_heartbeat": self.last_heartbeat,
        }
    
    def get_stats(self) -> dict:
        """Get current health statistics"""
        uptime = datetime.utcnow() - self.start_time
        return {
            "uptime_seconds": uptime.total_seconds(),
            "uptime_hours": uptime.total_seconds() / 3600,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "error_count": self.error_count,
            "restart_count": self.restart_count,
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "disk_percent": psutil.disk_usage('/').percent,
        }


class BotRunner:
    """Production bot runner with auto-restart and monitoring"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = None
        self.bot = None
        self.health_monitor = HealthMonitor()
        self.health_server: Optional[HealthCheckServer] = None
        self.running = False
        self.shutdown_requested = False
        
    async def initialize(self):
        """Initialize bot components"""
        logger.info("Initializing TrendCortex bot...")
        
        # Load configuration
        self.config = Config.load(self.config_path)
        logger.info(f"Configuration loaded from {self.config_path}")
        
        # Import main bot (deferred to avoid circular imports)
        from main import TrendCortexBot
        
        # Create bot instance with config dict
        self.bot = TrendCortexBot(self.config_path)
        await self.bot.initialize()
        
        # Start health check server
        if not self.health_server:
            self.health_server = HealthCheckServer(
                port=8080, 
                health_monitor=self.health_monitor,
                bot_instance=self.bot
            )
            self.health_server.start()
        
        logger.info("Bot initialized successfully")
        
    async def run_with_monitoring(self):
        """Run bot with health monitoring"""
        self.running = True
        
        # Create monitoring task
        monitor_task = asyncio.create_task(self._monitor_health())
        
        try:
            # Run the bot
            await self.bot.run()
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            logger.error(traceback.format_exc())
            self.health_monitor.record_error()
            raise
        finally:
            self.running = False
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_health(self):
        """Background task to monitor bot health"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update heartbeat
                self.health_monitor.update_heartbeat()
                
                # Check health
                is_healthy, reason = self.health_monitor.check_health()
                if not is_healthy:
                    logger.warning(f"Health check failed: {reason}")
                
                # Log health stats every 10 minutes
                if datetime.utcnow().minute % 10 == 0:
                    stats = self.health_monitor.get_stats()
                    logger.info(f"Health stats: {stats}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def run_with_auto_restart(self, max_restarts: int = 10):
        """
        Run bot with automatic restart on crashes.
        
        Args:
            max_restarts: Maximum number of consecutive restarts before giving up
        """
        restart_count = 0
        last_restart = datetime.utcnow()
        
        while not self.shutdown_requested and restart_count < max_restarts:
            try:
                # Reset restart counter if bot ran successfully for 1 hour
                if (datetime.utcnow() - last_restart).total_seconds() > 3600:
                    restart_count = 0
                
                # Initialize and run bot
                await self.initialize()
                logger.info(f"Starting bot (restart #{restart_count})")
                await self.run_with_monitoring()
                
                # If we get here, bot shut down gracefully
                logger.info("Bot shut down gracefully")
                break
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.shutdown_requested = True
                break
                
            except Exception as e:
                restart_count += 1
                last_restart = datetime.utcnow()
                self.health_monitor.record_restart()
                
                logger.error(f"Bot crashed: {e}")
                logger.error(traceback.format_exc())
                
                if restart_count < max_restarts:
                    wait_time = min(60 * restart_count, 300)  # Max 5 minutes
                    logger.info(f"Restarting in {wait_time} seconds... ({restart_count}/{max_restarts})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max restarts ({max_restarts}) reached, giving up")
                    break
            
            finally:
                # Cleanup
                if self.bot and hasattr(self.bot, 'shutdown'):
                    try:
                        await self.bot.shutdown()
                    except Exception as e:
                        logger.error(f"Cleanup error: {e}")
                    self.bot = None
        
        logger.info("Bot runner stopped")
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, requesting shutdown...")
        self.shutdown_requested = True
        if self.bot:
            self.bot.shutdown_requested = True


async def main():
    """Main entry point for production bot"""
    print("=" * 80)
    print("TrendCortex - Production Bot Runner")
    print("WEEX AI Wars - Alpha Awakens")
    print("=" * 80)
    print()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="TrendCortex Production Bot Runner")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--max-restarts", type=int, default=10, help="Max auto-restarts")
    args = parser.parse_args()
    
    # Create bot runner
    runner = BotRunner(config_path=args.config)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, runner.handle_signal)
    signal.signal(signal.SIGTERM, runner.handle_signal)
    
    # Run bot with auto-restart
    try:
        await runner.run_with_auto_restart(max_restarts=args.max_restarts)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("Bot runner exited")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
