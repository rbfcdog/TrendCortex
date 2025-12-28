#!/bin/bash

# TrendCortex Bot Management Script
# For managing the bot on Digital Ocean or any Linux server

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BOT_DIR="/opt/trendcortex"
SERVICE_NAME="trendcortex"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running with Docker or systemd
is_docker() {
    [ -f "${BOT_DIR}/docker-compose.yml" ] && command -v docker-compose &> /dev/null
}

# Status command
cmd_status() {
    log_info "Checking bot status..."
    echo ""
    
    if is_docker; then
        cd ${BOT_DIR}
        docker-compose ps
        echo ""
        log_info "Resource usage:"
        docker stats --no-stream trendcortex-bot
    else
        systemctl status ${SERVICE_NAME}
    fi
}

# Start command
cmd_start() {
    log_info "Starting bot..."
    
    if is_docker; then
        cd ${BOT_DIR}
        docker-compose up -d
        sleep 5
        docker-compose ps
    else
        sudo systemctl start ${SERVICE_NAME}
        sleep 2
        systemctl status ${SERVICE_NAME}
    fi
    
    log_info "Bot started successfully"
}

# Stop command
cmd_stop() {
    log_info "Stopping bot..."
    
    if is_docker; then
        cd ${BOT_DIR}
        docker-compose down
    else
        sudo systemctl stop ${SERVICE_NAME}
    fi
    
    log_info "Bot stopped"
}

# Restart command
cmd_restart() {
    log_info "Restarting bot..."
    cmd_stop
    sleep 2
    cmd_start
}

# Logs command
cmd_logs() {
    local lines=${1:-100}
    local follow=${2:-false}
    
    log_info "Viewing logs (last ${lines} lines)..."
    echo ""
    
    if is_docker; then
        cd ${BOT_DIR}
        if [ "$follow" = "true" ]; then
            docker-compose logs -f --tail=${lines}
        else
            docker-compose logs --tail=${lines}
        fi
    else
        if [ "$follow" = "true" ]; then
            sudo journalctl -u ${SERVICE_NAME} -n ${lines} -f
        else
            sudo journalctl -u ${SERVICE_NAME} -n ${lines}
        fi
    fi
}

# Health command
cmd_health() {
    log_info "Checking bot health..."
    echo ""
    
    # Check if bot is running
    if is_docker; then
        if ! docker ps | grep -q trendcortex-bot; then
            log_error "Bot container is not running!"
            return 1
        fi
    else
        if ! systemctl is-active --quiet ${SERVICE_NAME}; then
            log_error "Bot service is not active!"
            return 1
        fi
    fi
    
    # Check recent errors
    log_info "Checking for recent errors..."
    if is_docker; then
        error_count=$(cd ${BOT_DIR} && docker-compose logs --tail=100 | grep -i error | wc -l)
    else
        error_count=$(sudo journalctl -u ${SERVICE_NAME} -n 100 | grep -i error | wc -l)
    fi
    
    if [ $error_count -gt 10 ]; then
        log_warn "Found ${error_count} errors in last 100 log lines"
    else
        log_info "Error count: ${error_count} (OK)"
    fi
    
    # Check system resources
    log_info "System resources:"
    echo ""
    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%"
    echo "Memory Usage: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')"
    echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"
    echo ""
    
    # Check log file sizes
    log_info "Log directory size: $(du -sh ${BOT_DIR}/logs 2>/dev/null || echo 'N/A')"
    
    log_info "Health check complete"
}

# Update command
cmd_update() {
    log_info "Updating bot code..."
    
    cd ${BOT_DIR}
    
    # Backup current code
    log_info "Creating backup..."
    tar -czf backup-$(date +%Y%m%d-%H%M%S).tar.gz \
        --exclude='logs' \
        --exclude='data' \
        --exclude='models' \
        --exclude='*.tar.gz' \
        .
    
    # Pull latest code (if using git)
    if [ -d ".git" ]; then
        log_info "Pulling latest code from git..."
        git pull
    else
        log_warn "Not a git repository, manual update required"
    fi
    
    # Rebuild if Docker
    if is_docker; then
        log_info "Rebuilding Docker image..."
        docker-compose build
    else
        log_info "Reinstalling dependencies..."
        ${BOT_DIR}/venv/bin/pip install -r requirements.txt
    fi
    
    # Restart bot
    cmd_restart
    
    log_info "Update complete"
}

# Backup command
cmd_backup() {
    local backup_dir="${BOT_DIR}/backups"
    mkdir -p ${backup_dir}
    
    local backup_file="${backup_dir}/backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    log_info "Creating backup..."
    cd ${BOT_DIR}
    tar -czf ${backup_file} \
        --exclude='backups' \
        --exclude='*.tar.gz' \
        config.json logs/ data/ models/
    
    log_info "Backup created: ${backup_file}"
    log_info "Backup size: $(du -h ${backup_file} | cut -f1)"
    
    # Keep only last 10 backups
    ls -t ${backup_dir}/backup-*.tar.gz | tail -n +11 | xargs -r rm
    log_info "Kept last 10 backups, removed older ones"
}

# Monitor command
cmd_monitor() {
    log_info "Starting real-time monitoring (Ctrl+C to exit)..."
    echo ""
    
    while true; do
        clear
        echo "=========================================="
        echo "TrendCortex Bot Monitor"
        echo "Time: $(date)"
        echo "=========================================="
        echo ""
        
        # Bot status
        if is_docker; then
            echo "Docker Status:"
            docker ps --filter name=trendcortex-bot --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            echo ""
            
            echo "Resource Usage:"
            docker stats --no-stream trendcortex-bot --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        else
            echo "Service Status:"
            systemctl status ${SERVICE_NAME} --no-pager | head -n 5
            echo ""
        fi
        
        echo ""
        echo "System Resources:"
        echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%"
        echo "  Memory: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')"
        echo "  Disk: $(df -h / | awk 'NR==2{print $5}')"
        echo ""
        
        echo "Recent Logs (last 10 lines):"
        echo "------------------------------------------"
        if is_docker; then
            cd ${BOT_DIR} && docker-compose logs --tail=10 | tail -n 10
        else
            sudo journalctl -u ${SERVICE_NAME} -n 10 --no-pager
        fi
        
        echo ""
        echo "Press Ctrl+C to exit"
        sleep 10
    done
}

# Clean command
cmd_clean() {
    log_warn "Cleaning old logs and cache..."
    
    # Clean old logs (keep last 7 days)
    find ${BOT_DIR}/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Clean cache
    find ${BOT_DIR}/data/cache -type f -mtime +1 -delete 2>/dev/null || true
    
    # Clean old backups (keep last 10)
    ls -t ${BOT_DIR}/backups/backup-*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm
    
    log_info "Cleanup complete"
}

# Performance stats
cmd_stats() {
    log_info "Performance Statistics"
    echo ""
    
    echo "=== Trading Performance ==="
    if [ -f "${BOT_DIR}/logs/executions/executions.log" ]; then
        echo "Total Trades: $(grep -c "status.*FILLED" ${BOT_DIR}/logs/executions/executions.log 2>/dev/null || echo 0)"
        echo "Last Trade: $(tail -n 1 ${BOT_DIR}/logs/executions/executions.log 2>/dev/null | jq -r '.timestamp' 2>/dev/null || echo 'N/A')"
    else
        echo "No trading data available"
    fi
    
    echo ""
    echo "=== System Uptime ==="
    if is_docker; then
        docker ps --filter name=trendcortex-bot --format "{{.Status}}"
    else
        systemctl show ${SERVICE_NAME} -p ActiveEnterTimestamp --value
    fi
    
    echo ""
    echo "=== Resource Usage ==="
    if is_docker; then
        docker stats --no-stream trendcortex-bot --format "CPU: {{.CPUPerc}}, Memory: {{.MemPerc}}"
    fi
    
    echo ""
    echo "=== Log Statistics ==="
    echo "Signals: $(find ${BOT_DIR}/logs/signals -name "*.log" 2>/dev/null | wc -l) files"
    echo "Decisions: $(find ${BOT_DIR}/logs/decisions -name "*.log" 2>/dev/null | wc -l) files"
    echo "Executions: $(find ${BOT_DIR}/logs/executions -name "*.log" 2>/dev/null | wc -l) files"
    echo "Total log size: $(du -sh ${BOT_DIR}/logs 2>/dev/null | cut -f1)"
}

# Main menu
show_menu() {
    echo ""
    echo "=========================================="
    echo "TrendCortex Bot Management"
    echo "=========================================="
    echo ""
    echo "1) Status     - Check bot status"
    echo "2) Start      - Start the bot"
    echo "3) Stop       - Stop the bot"
    echo "4) Restart    - Restart the bot"
    echo "5) Logs       - View recent logs"
    echo "6) Follow     - Follow logs in real-time"
    echo "7) Health     - Run health check"
    echo "8) Monitor    - Real-time monitoring"
    echo "9) Stats      - Performance statistics"
    echo "10) Update    - Update bot code"
    echo "11) Backup    - Create backup"
    echo "12) Clean     - Clean old logs"
    echo "0) Exit"
    echo ""
}

# Main script
main() {
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice
            echo ""
            
            case $choice in
                1) cmd_status ;;
                2) cmd_start ;;
                3) cmd_stop ;;
                4) cmd_restart ;;
                5) cmd_logs 100 false ;;
                6) cmd_logs 100 true ;;
                7) cmd_health ;;
                8) cmd_monitor ;;
                9) cmd_stats ;;
                10) cmd_update ;;
                11) cmd_backup ;;
                12) cmd_clean ;;
                0) echo "Goodbye!"; exit 0 ;;
                *) log_error "Invalid option" ;;
            esac
            
            echo ""
            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            status) cmd_status ;;
            start) cmd_start ;;
            stop) cmd_stop ;;
            restart) cmd_restart ;;
            logs) cmd_logs ${2:-100} false ;;
            follow) cmd_logs ${2:-100} true ;;
            health) cmd_health ;;
            monitor) cmd_monitor ;;
            stats) cmd_stats ;;
            update) cmd_update ;;
            backup) cmd_backup ;;
            clean) cmd_clean ;;
            *)
                echo "Usage: $0 {status|start|stop|restart|logs|follow|health|monitor|stats|update|backup|clean}"
                exit 1
                ;;
        esac
    fi
}

main "$@"
