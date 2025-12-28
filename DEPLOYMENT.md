# TrendCortex Deployment Guide

Complete guide for deploying TrendCortex bot on Digital Ocean for 24/7 operation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Docker Deployment](#docker-deployment)
4. [Systemd Deployment](#systemd-deployment)
5. [Digital Ocean Setup](#digital-ocean-setup)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

---

## Prerequisites

### Local Machine

- **doctl** (Digital Ocean CLI)
  ```bash
  # Install doctl
  cd ~
  wget https://github.com/digitalocean/doctl/releases/download/v1.98.0/doctl-1.98.0-linux-amd64.tar.gz
  tar xf doctl-1.98.0-linux-amd64.tar.gz
  sudo mv doctl /usr/local/bin
  
  # Authenticate
  doctl auth init
  ```

- **SSH Key** registered with Digital Ocean
  ```bash
  # Generate if needed
  ssh-keygen -t ed25519 -C "your_email@example.com"
  
  # Add to Digital Ocean
  doctl compute ssh-key import trendcortex-key --public-key-file ~/.ssh/id_ed25519.pub
  ```

### WEEX API Credentials

You need:
- API Key
- API Secret
- User ID (for AI logging)

Get them from [WEEX Exchange](https://www.weex.com)

---

## Quick Start

### 1. Configure Your Bot

Create `config.json`:

```json
{
  "api": {
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET",
    "base_url": "https://api.weex.com"
  },
  "ai_logger": {
    "enabled": true,
    "user_id": "YOUR_USER_ID",
    "endpoint": "https://ai-wars-api.weex.com/api/log"
  },
  "trading": {
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "max_position_size": 0.02,
    "risk_per_trade": 0.01
  },
  "intervals": {
    "data_fetch": 60,
    "signal_check": 30,
    "health_check": 60
  }
}
```

### 2. Deploy to Digital Ocean

```bash
# Automated deployment
chmod +x deploy-digital-ocean.sh
./deploy-digital-ocean.sh

# Or with custom settings
DROPLET_NAME="my-trendcortex" \
DROPLET_SIZE="s-4vcpu-8gb" \
REGION="nyc1" \
./deploy-digital-ocean.sh
```

### 3. Manage Your Bot

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Check status
cd /opt/trendcortex
docker-compose ps

# View logs
docker-compose logs -f

# Or use management script
./bot-manage.sh status
```

---

## Docker Deployment

### Initial Setup

1. **Build and start**:
   ```bash
   docker-compose up -d
   ```

2. **Check status**:
   ```bash
   docker-compose ps
   docker-compose logs
   ```

3. **View specific logs**:
   ```bash
   docker-compose logs trendcortex -f
   ```

### With Monitoring

Enable Prometheus and Grafana:

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access Grafana
# http://YOUR_SERVER_IP:3000
# Default: admin/admin
```

### Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# Rebuild
docker-compose build --no-cache
docker-compose up -d

# View logs
docker-compose logs -f --tail=100

# Execute commands in container
docker-compose exec trendcortex bash

# Check resource usage
docker stats trendcortex-bot
```

### Volume Management

Persistent data is stored in:
- `./logs/` - All log files
- `./data/` - Market data cache
- `./models/` - ML model files

```bash
# Backup volumes
tar -czf backup-$(date +%Y%m%d).tar.gz logs/ data/ models/ config.json

# Restore volumes
tar -xzf backup-YYYYMMDD.tar.gz
```

---

## Systemd Deployment

Alternative to Docker for direct system installation.

### Setup

1. **Install dependencies**:
   ```bash
   # System packages
   sudo apt-get update
   sudo apt-get install -y python3.11 python3.11-venv python3-pip

   # Create virtual environment
   cd /opt/trendcortex
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install service**:
   ```bash
   # Copy service file
   sudo cp deployment/trendcortex.service /etc/systemd/system/

   # Reload systemd
   sudo systemctl daemon-reload

   # Enable service
   sudo systemctl enable trendcortex

   # Start service
   sudo systemctl start trendcortex
   ```

3. **Verify**:
   ```bash
   sudo systemctl status trendcortex
   sudo journalctl -u trendcortex -f
   ```

### Systemd Commands

```bash
# Status
sudo systemctl status trendcortex

# Start
sudo systemctl start trendcortex

# Stop
sudo systemctl stop trendcortex

# Restart
sudo systemctl restart trendcortex

# View logs
sudo journalctl -u trendcortex -f
sudo journalctl -u trendcortex -n 100

# Enable on boot
sudo systemctl enable trendcortex

# Disable on boot
sudo systemctl disable trendcortex
```

---

## Digital Ocean Setup

### Manual Droplet Creation

1. **Create droplet**:
   ```bash
   doctl compute droplet create trendcortex-bot \
     --size s-2vcpu-4gb \
     --image docker-20-04 \
     --region nyc1 \
     --ssh-keys $(doctl compute ssh-key list --format ID --no-header | head -n 1) \
     --wait
   ```

2. **Get droplet IP**:
   ```bash
   DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep trendcortex-bot | awk '{print $2}')
   echo $DROPLET_IP
   ```

3. **Upload files**:
   ```bash
   # Create directory
   ssh root@$DROPLET_IP "mkdir -p /opt/trendcortex"
   
   # Upload project
   scp -r ./* root@$DROPLET_IP:/opt/trendcortex/
   ```

4. **Start bot**:
   ```bash
   ssh root@$DROPLET_IP "cd /opt/trendcortex && docker-compose up -d"
   ```

### Droplet Sizing

Recommended sizes based on trading intensity:

| Droplet Size | vCPUs | RAM | Price/mo | Use Case |
|--------------|-------|-----|----------|----------|
| s-2vcpu-4gb | 2 | 4GB | $24 | Light trading (1-2 pairs) |
| s-4vcpu-8gb | 4 | 8GB | $48 | Medium trading (3-5 pairs) |
| s-8vcpu-16gb | 8 | 16GB | $96 | Heavy trading (5+ pairs, ML models) |

### Firewall Setup

```bash
# Create firewall
doctl compute firewall create \
  --name trendcortex-fw \
  --inbound-rules "protocol:tcp,ports:22,sources:addresses:0.0.0.0/0" \
  --inbound-rules "protocol:tcp,ports:3000,sources:addresses:0.0.0.0/0" \
  --inbound-rules "protocol:tcp,ports:9090,sources:addresses:0.0.0.0/0" \
  --outbound-rules "protocol:tcp,ports:all,destinations:addresses:0.0.0.0/0"

# Apply to droplet
doctl compute firewall add-droplets FIREWALL_ID --droplet-ids DROPLET_ID
```

---

## Configuration

### Environment Variables

For sensitive data, use environment variables instead of config file:

```bash
# Docker
cat > .env << EOF
WEEX_API_KEY=your_api_key
WEEX_API_SECRET=your_api_secret
WEEX_USER_ID=your_user_id
EOF

docker-compose up -d
```

```bash
# Systemd
sudo systemctl edit trendcortex

# Add:
[Service]
Environment="WEEX_API_KEY=your_api_key"
Environment="WEEX_API_SECRET=your_api_secret"
Environment="WEEX_USER_ID=your_user_id"
```

### Config File Sections

#### API Settings

```json
{
  "api": {
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET",
    "base_url": "https://api.weex.com",
    "timeout": 30,
    "rate_limit": {
      "requests_per_second": 10,
      "burst": 20
    }
  }
}
```

#### Trading Settings

```json
{
  "trading": {
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "max_position_size": 0.02,
    "risk_per_trade": 0.01,
    "max_open_positions": 3,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.05
  }
}
```

#### ML Model Settings

```json
{
  "models": {
    "transformer": {
      "enabled": true,
      "model_path": "models/transformer_v1.pth",
      "sequence_length": 100,
      "confidence_threshold": 0.7
    },
    "llm": {
      "enabled": true,
      "provider": "openai",
      "model": "gpt-4",
      "temperature": 0.3
    }
  }
}
```

#### AI Logging

```json
{
  "ai_logger": {
    "enabled": true,
    "user_id": "YOUR_USER_ID",
    "endpoint": "https://ai-wars-api.weex.com/api/log",
    "batch_size": 10,
    "flush_interval": 60,
    "retry_attempts": 3
  }
}
```

---

## Monitoring

### Built-in Health Checks

The bot includes health monitoring:

```python
# Health metrics tracked:
# - Last heartbeat time
# - Error rate (errors per minute)
# - Restart count
# - CPU usage
# - Memory usage
# - Disk usage
```

View health status:
```bash
# Docker
docker inspect --format='{{.State.Health.Status}}' trendcortex-bot

# Management script
./bot-manage.sh health
```

### Log Monitoring

Logs are organized by type:

```
logs/
├── main.log                    # Main bot logs
├── signals/                    # Trading signals
│   └── YYYYMMDD.log
├── decisions/                  # Trading decisions
│   └── YYYYMMDD.log
├── executions/                 # Order executions
│   └── YYYYMMDD.log
└── ai_logger/                  # AI logging activity
    └── YYYYMMDD.log
```

Monitor in real-time:
```bash
# All logs
docker-compose logs -f

# Specific log type
tail -f logs/executions/$(date +%Y%m%d).log

# Errors only
docker-compose logs -f | grep ERROR

# Real-time monitoring
./bot-manage.sh monitor
```

### Prometheus + Grafana

1. **Start monitoring stack**:
   ```bash
   docker-compose --profile monitoring up -d
   ```

2. **Access Grafana**:
   - URL: `http://YOUR_SERVER_IP:3000`
   - Default credentials: `admin/admin`

3. **Add Prometheus data source**:
   - Settings → Data Sources → Add Prometheus
   - URL: `http://prometheus:9090`

4. **Import dashboard**:
   - Create → Import → Upload `monitoring/grafana-dashboard.json`

### Metrics Exported

- **Trading Metrics**:
  - Total trades
  - Win rate
  - Profit/Loss
  - Open positions
  
- **System Metrics**:
  - CPU usage
  - Memory usage
  - API latency
  - Error rate

- **Bot Metrics**:
  - Uptime
  - Restarts
  - Last signal time
  - Last trade time

### Alerts

Set up alerts for critical events:

```yaml
# monitoring/alerts.yml
groups:
  - name: trendcortex
    interval: 1m
    rules:
      - alert: BotDown
        expr: up{job="trendcortex"} == 0
        for: 5m
        
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
        for: 5m
        
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 90
        for: 5m
```

---

## Troubleshooting

### Bot Won't Start

1. **Check logs**:
   ```bash
   docker-compose logs trendcortex
   # or
   sudo journalctl -u trendcortex -n 100
   ```

2. **Validate config**:
   ```bash
   # Test config file
   python3 -c "import json; json.load(open('config.json'))"
   ```

3. **Check API credentials**:
   ```bash
   # Test API connection
   docker-compose exec trendcortex python -c "
   from trendcortex.api_client import WEEXAPIClient
   import json
   config = json.load(open('config.json'))
   client = WEEXAPIClient(config['api'])
   print(client.get_account_balance())
   "
   ```

### High Memory Usage

1. **Check resource usage**:
   ```bash
   docker stats trendcortex-bot
   ```

2. **Reduce memory footprint**:
   ```json
   {
     "trading": {
       "symbols": ["BTC/USDT"],  // Reduce pairs
       "data_lookback": 100       // Reduce history
     },
     "models": {
       "transformer": {
         "enabled": false         // Disable heavy models
       }
     }
   }
   ```

3. **Increase droplet RAM** or restart bot periodically:
   ```bash
   # Add to crontab
   0 */6 * * * docker-compose -f /opt/trendcortex/docker-compose.yml restart
   ```

### API Errors

1. **Check rate limits**:
   ```bash
   grep "rate limit" logs/main.log
   ```

2. **Adjust rate limiting**:
   ```json
   {
     "api": {
       "rate_limit": {
         "requests_per_second": 5,  // Reduce RPS
         "burst": 10
       }
     }
   }
   ```

3. **Verify API key permissions**:
   - Go to WEEX → API Management
   - Ensure trading permissions enabled

### Connection Issues

1. **Check network**:
   ```bash
   docker-compose exec trendcortex ping -c 3 api.weex.com
   ```

2. **Check firewall**:
   ```bash
   sudo iptables -L -n
   ```

3. **Enable debug logging**:
   ```json
   {
     "logging": {
       "level": "DEBUG",
       "log_api_requests": true
     }
   }
   ```

### Bot Keeps Restarting

1. **Check restart count**:
   ```bash
   ./bot-manage.sh health
   ```

2. **View crash logs**:
   ```bash
   docker-compose logs --tail=500 trendcortex | grep -A 20 "ERROR\|CRITICAL"
   ```

3. **Disable auto-restart temporarily**:
   ```yaml
   # docker-compose.yml
   services:
     trendcortex:
       restart: "no"  # Change from "always"
   ```

### AI Logging Failures

1. **Check AI logger logs**:
   ```bash
   cat logs/ai_logger/$(date +%Y%m%d).log
   ```

2. **Test AI endpoint**:
   ```bash
   curl -X POST https://ai-wars-api.weex.com/api/log \
     -H "Content-Type: application/json" \
     -d '{"user_id":"test","timestamp":"2024-01-01T00:00:00Z"}'
   ```

3. **Disable temporarily**:
   ```json
   {
     "ai_logger": {
       "enabled": false
     }
   }
   ```

---

## Maintenance

### Daily Tasks

Run health check:
```bash
./bot-manage.sh health
```

### Weekly Tasks

1. **Review performance**:
   ```bash
   ./bot-manage.sh stats
   ```

2. **Check log sizes**:
   ```bash
   du -sh logs/
   ```

3. **Clean old logs**:
   ```bash
   ./bot-manage.sh clean
   ```

### Monthly Tasks

1. **Update bot**:
   ```bash
   ./bot-manage.sh update
   ```

2. **Backup data**:
   ```bash
   ./bot-manage.sh backup
   ```

3. **Review costs**:
   ```bash
   doctl compute droplet list
   doctl billing balance
   ```

### Backup Strategy

**Automated backups** (add to crontab):

```bash
# Daily backup at 2 AM
0 2 * * * cd /opt/trendcortex && ./bot-manage.sh backup

# Weekly backup to remote storage
0 3 * * 0 cd /opt/trendcortex && \
  tar -czf - logs/ data/ models/ config.json | \
  ssh backup-server "cat > /backups/trendcortex-$(date +\%Y\%m\%d).tar.gz"
```

**Manual backup**:
```bash
./bot-manage.sh backup
```

**Restore from backup**:
```bash
tar -xzf backups/backup-YYYYMMDD-HHMMSS.tar.gz
docker-compose restart
```

### Updating

1. **Backup first**:
   ```bash
   ./bot-manage.sh backup
   ```

2. **Pull changes**:
   ```bash
   git pull
   ```

3. **Rebuild**:
   ```bash
   docker-compose build --no-cache
   docker-compose down
   docker-compose up -d
   ```

4. **Verify**:
   ```bash
   docker-compose logs -f --tail=100
   ./bot-manage.sh health
   ```

### Scaling

**Horizontal scaling** (multiple bots):

1. Create separate directories:
   ```bash
   cp -r /opt/trendcortex /opt/trendcortex-2
   cd /opt/trendcortex-2
   ```

2. Modify config (different symbols):
   ```json
   {
     "trading": {
       "symbols": ["SOL/USDT", "ADA/USDT"]
     }
   }
   ```

3. Change ports in docker-compose.yml:
   ```yaml
   ports:
     - "8081:8080"  # Different port
   ```

4. Start second instance:
   ```bash
   docker-compose up -d
   ```

**Vertical scaling** (more resources):

```bash
# Resize droplet
doctl compute droplet-action resize DROPLET_ID --size s-4vcpu-8gb --wait

# Increase resource limits in docker-compose.yml
docker-compose down
docker-compose up -d
```

---

## Security Best Practices

1. **API Keys**:
   - Store in environment variables, not config files
   - Use read-only keys for testing
   - Rotate keys monthly

2. **Server Access**:
   - Disable password authentication
   - Use SSH keys only
   - Enable firewall
   - Keep system updated

3. **Network Security**:
   ```bash
   # Minimal firewall rules
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow ssh
   sudo ufw enable
   ```

4. **Log Security**:
   - Don't log API secrets
   - Encrypt backups
   - Use secure backup storage

5. **Docker Security**:
   - Run as non-root user (already configured)
   - Use read-only volumes where possible
   - Scan images for vulnerabilities:
     ```bash
     docker scan trendcortex:latest
     ```

---

## Support

### Getting Help

1. **Check logs first**:
   ```bash
   ./bot-manage.sh logs 500
   ```

2. **Run diagnostics**:
   ```bash
   ./bot-manage.sh health
   ```

3. **Review documentation**:
   - This deployment guide
   - `docs/AI_LOGGING.md` for AI logging issues
   - `README.md` for general bot usage

### Common Resources

- WEEX API Docs: https://www.weex.com/api-docs
- Digital Ocean Docs: https://docs.digitalocean.com
- Docker Docs: https://docs.docker.com

### Reporting Issues

When reporting issues, include:
- Bot version
- Deployment method (Docker/Systemd)
- Error logs (last 100 lines)
- System info (CPU, RAM, OS)
- Configuration (redact secrets)

---

## Quick Reference

### Essential Commands

```bash
# Status
./bot-manage.sh status

# Logs
./bot-manage.sh logs 100
./bot-manage.sh follow

# Control
./bot-manage.sh start
./bot-manage.sh stop
./bot-manage.sh restart

# Health
./bot-manage.sh health
./bot-manage.sh monitor

# Maintenance
./bot-manage.sh backup
./bot-manage.sh clean
./bot-manage.sh update
```

### File Locations

- **Config**: `/opt/trendcortex/config.json`
- **Logs**: `/opt/trendcortex/logs/`
- **Data**: `/opt/trendcortex/data/`
- **Models**: `/opt/trendcortex/models/`
- **Backups**: `/opt/trendcortex/backups/`

### Important URLs

- **WEEX Exchange**: https://www.weex.com
- **AI Wars API**: https://ai-wars-api.weex.com
- **Grafana** (if enabled): http://YOUR_SERVER:3000
- **Prometheus** (if enabled): http://YOUR_SERVER:9090

---

## Next Steps

After deployment:

1. ✅ Verify bot is running: `./bot-manage.sh status`
2. ✅ Check AI logging: `cat logs/ai_logger/$(date +%Y%m%d).log`
3. ✅ Monitor first trades: `./bot-manage.sh follow`
4. ✅ Set up monitoring: `docker-compose --profile monitoring up -d`
5. ✅ Schedule backups: Add cron jobs
6. ✅ Review performance: Check after 24 hours

**Your bot is now running 24/7! Good luck with the WEEX AI Wars!** 🚀
