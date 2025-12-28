"""
Health Check HTTP Server

Provides health status and metrics endpoint for monitoring
"""

import asyncio
import json
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check and metrics endpoints"""
    
    # Class variable to store health monitor reference
    health_monitor: Optional[Any] = None
    bot_instance: Optional[Any] = None
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.handle_health()
        elif self.path == '/metrics':
            self.handle_metrics()
        elif self.path == '/status':
            self.handle_status()
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_health(self):
        """Health check endpoint for Docker/Kubernetes"""
        try:
            if not self.health_monitor:
                self.send_json_response({"status": "unknown"}, 503)
                return
            
            is_healthy = self.health_monitor.is_healthy()
            status_code = 200 if is_healthy else 503
            
            response = {
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.health_monitor.start_time).total_seconds()
            }
            
            self.send_json_response(response, status_code)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.send_json_response({"status": "error", "error": str(e)}, 500)
    
    def handle_metrics(self):
        """Prometheus-style metrics endpoint"""
        try:
            if not self.health_monitor:
                self.send_error(503, "Health monitor not available")
                return
            
            metrics = self.health_monitor.get_metrics()
            
            # Convert to Prometheus text format
            lines = [
                "# HELP trendcortex_up Bot is running",
                "# TYPE trendcortex_up gauge",
                f"trendcortex_up 1",
                "",
                "# HELP trendcortex_restart_count Total number of restarts",
                "# TYPE trendcortex_restart_count counter",
                f"trendcortex_restart_count {metrics['restart_count']}",
                "",
                "# HELP trendcortex_errors_total Total number of errors",
                "# TYPE trendcortex_errors_total counter",
                f"trendcortex_errors_total {metrics['error_count']}",
                "",
                "# HELP trendcortex_cpu_usage_percent CPU usage percentage",
                "# TYPE trendcortex_cpu_usage_percent gauge",
                f"trendcortex_cpu_usage_percent {metrics['cpu_percent']}",
                "",
                "# HELP trendcortex_memory_usage_percent Memory usage percentage",
                "# TYPE trendcortex_memory_usage_percent gauge",
                f"trendcortex_memory_usage_percent {metrics['memory_percent']}",
                "",
                "# HELP trendcortex_disk_usage_percent Disk usage percentage",
                "# TYPE trendcortex_disk_usage_percent gauge",
                f"trendcortex_disk_usage_percent {metrics['disk_percent']}",
                "",
            ]
            
            # Add last heartbeat
            if metrics['last_heartbeat']:
                seconds_since = (datetime.now() - metrics['last_heartbeat']).total_seconds()
                lines.extend([
                    "# HELP trendcortex_last_heartbeat_seconds Seconds since last heartbeat",
                    "# TYPE trendcortex_last_heartbeat_seconds gauge",
                    f"trendcortex_last_heartbeat_seconds {seconds_since}",
                    ""
                ])
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write('\n'.join(lines).encode())
            
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            self.send_error(500, str(e))
    
    def handle_status(self):
        """Detailed status endpoint"""
        try:
            if not self.health_monitor:
                self.send_json_response({"status": "unknown"}, 503)
                return
            
            metrics = self.health_monitor.get_metrics()
            
            status = {
                "bot": {
                    "name": "TrendCortex",
                    "status": "running" if self.health_monitor.is_healthy() else "unhealthy",
                    "uptime_seconds": (datetime.now() - self.health_monitor.start_time).total_seconds(),
                    "restart_count": metrics['restart_count'],
                    "last_restart": metrics.get('last_restart_time')
                },
                "health": {
                    "is_healthy": self.health_monitor.is_healthy(),
                    "last_heartbeat": metrics['last_heartbeat'].isoformat() if metrics['last_heartbeat'] else None,
                    "error_rate": metrics['error_rate'],
                    "consecutive_errors": metrics['error_count']
                },
                "system": {
                    "cpu_percent": metrics['cpu_percent'],
                    "memory_percent": metrics['memory_percent'],
                    "disk_percent": metrics['disk_percent']
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add bot-specific stats if available
            if self.bot_instance and hasattr(self.bot_instance, 'get_stats'):
                try:
                    bot_stats = self.bot_instance.get_stats()
                    status['trading'] = bot_stats
                except Exception as e:
                    logger.warning(f"Could not get bot stats: {e}")
            
            self.send_json_response(status, 200)
            
        except Exception as e:
            logger.error(f"Status error: {e}")
            self.send_json_response({"status": "error", "error": str(e)}, 500)
    
    def send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        # Only log errors
        if args[1] != '200':
            logger.debug(f"Health check: {format % args}")


class HealthCheckServer:
    """HTTP server for health checks and metrics"""
    
    def __init__(self, port: int = 8080, health_monitor=None, bot_instance=None):
        self.port = port
        self.health_monitor = health_monitor
        self.bot_instance = bot_instance
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[Thread] = None
        
        # Set class variables for handler
        HealthCheckHandler.health_monitor = health_monitor
        HealthCheckHandler.bot_instance = bot_instance
    
    def start(self):
        """Start the HTTP server in a background thread"""
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), HealthCheckHandler)
            self.thread = Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"Health check server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")
    
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            logger.info("Health check server stopped")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock health monitor for testing
    class MockHealthMonitor:
        def __init__(self):
            self.start_time = datetime.now()
        
        def is_healthy(self):
            return True
        
        def get_metrics(self):
            return {
                'restart_count': 0,
                'error_count': 0,
                'error_rate': 0.0,
                'cpu_percent': 25.5,
                'memory_percent': 45.2,
                'disk_percent': 60.3,
                'last_heartbeat': datetime.now()
            }
    
    monitor = MockHealthMonitor()
    server = HealthCheckServer(port=8080, health_monitor=monitor)
    server.start()
    
    print("Health check server running on http://localhost:8080")
    print("Endpoints:")
    print("  /health  - Health check (for Docker)")
    print("  /metrics - Prometheus metrics")
    print("  /status  - Detailed status JSON")
    print("\nPress Ctrl+C to stop")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        server.stop()
