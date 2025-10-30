#!/bin/bash

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Переменные
PROJECT_NAME="blackfox"
PROJECT_ROOT="/root/$PROJECT_NAME"
APP_DIR="$PROJECT_ROOT/app"
SERVER_RUN_DIR="$APP_DIR/server_run"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
SERVICE_USER="root"

# Проверка прав
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Этот скрипт должен запускаться с правами root"
        exit 1
    fi
}

# Проверка структуры проекта
check_directories() {
    log_info "Проверка структуры проекта..."
    
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        log_error "Основная директория проекта не найдена: $PROJECT_ROOT"
        exit 1
    fi
    
    if [[ ! -f "$SERVER_RUN_DIR/main.py" ]]; then
        log_error "Основной файл приложения не найден: $SERVER_RUN_DIR/main.py"
        exit 1
    fi
    
    log_success "Структура проекта проверена"
}

# Поиск virtualenv Poetry
find_poetry_venv() {
    log_info "Поиск virtualenv Poetry..."
    
    # Пробуем получить путь через poetry
    if command -v poetry &> /dev/null; then
        cd "$PROJECT_ROOT"
        local poetry_venv=$(poetry env info --path 2>/dev/null || true)
        if [[ -n "$poetry_venv" && -f "$poetry_venv/bin/activate" ]]; then
            echo "$poetry_venv"
            log_success "Найден virtualenv через poetry: $poetry_venv"
            return 0
        fi
    fi
    
    # Поиск вручную
    local possible_paths=(
        "/root/.cache/pypoetry/virtualenvs/blackfox-*"
        "/root/.cache/pypoetry/virtualenvs/blackfox-FAJo6lti-py3.12"
        "$PROJECT_ROOT/.venv"
    )
    
    for pattern in "${possible_paths[@]}"; do
        for path in $pattern; do
            if [[ -d "$path" && -f "$path/bin/activate" ]]; then
                echo "$path"
                log_success "Найден virtualenv: $path"
                return 0
            fi
        done
    done
    
    log_error "Virtualenv Poetry не найден!"
    log_info "Доступные виртуальные окружения:"
    ls -la /root/.cache/pypoetry/virtualenvs/ 2>/dev/null || echo "Директория virtualenvs не существует"
    log_info "Выполните: cd $PROJECT_ROOT && poetry install"
    exit 1
}

# Создание systemd сервиса для API
create_api_service() {
    local venv_path="$1"
    local service_file="/etc/systemd/system/$PROJECT_NAME-api.service"
    
    log_info "Создание systemd сервиса для API..."
    
    # Проверяем существующий сервис
    if [[ -f "$service_file" ]]; then
        log_warning "Сервис уже существует, создаем резервную копию..."
        cp "$service_file" "$service_file.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    cat > "$service_file" << EOF
[Unit]
Description=BlackFox FastAPI ML Application
After=network.target
Wants=network.target

[Service]
Type=exec
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
Environment="PYTHONPATH=$PROJECT_ROOT"
Environment="PATH=$venv_path/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ExecStart=$venv_path/bin/uvicorn app.server_run.main:app \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --workers 2

# Перезагрузка при сбое
Restart=always
RestartSec=5

# Лимиты ресурсов
LimitNOFILE=65536
LimitNPROC=4096

# Безопасность
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_ROOT
ReadWritePaths=/tmp

# Логирование
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$PROJECT_NAME-api

[Install]
WantedBy=multi-user.target
EOF

    log_success "Создан systemd сервис: $service_file"
}

# Упрощенная настройка Nginx (опционально)
setup_nginx_optional() {
    log_info "Проверка Nginx..."
    
    if ! command -v nginx &> /dev/null; then
        log_warning "Nginx не установлен. Пропускаем настройку."
        return 0
    fi
    
    local nginx_dir="/etc/nginx/sites-available"
    local nginx_enabled_dir="/etc/nginx/sites-enabled"
    local config_file="$nginx_dir/$PROJECT_NAME"
    
    # Создаем простую HTTP конфигурацию
    cat > "$config_file" << EOF
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /health {
        proxy_pass http://localhost:8000/health;
        proxy_set_header Host \$host;
        access_log off;
    }
    
    location /metrics {
        proxy_pass http://localhost:8000/metrics;
        proxy_set_header Host \$host;
        access_log off;
    }
}
EOF

    # Активация конфига
    ln -sf "$config_file" "$nginx_enabled_dir/$PROJECT_NAME"
    
    # Отключаем дефолтный конфиг
    if [[ -f "$nginx_enabled_dir/default" ]]; then
        rm -f "$nginx_enabled_dir/default"
    fi
    
    # Проверка конфигурации
    if nginx -t &> /dev/null; then
        log_success "Nginx конфигурация проверена"
        systemctl reload nginx
    else
        log_warning "Ошибка в конфигурации Nginx, отключаем..."
        rm -f "$nginx_enabled_dir/$PROJECT_NAME"
        systemctl reload nginx
    fi
}

# Создание скриптов управления
create_management_scripts() {
    log_info "Создание скриптов управления..."
    
    # Основной скрипт управления
    cat > "$SCRIPTS_DIR/manage.sh" << 'EOF'
#!/bin/bash

set -e

PROJECT_NAME="blackfox"
SERVICE_API="$PROJECT_NAME-api.service"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    echo "Управление сервисом BlackFox API"
    echo "Использование: $0 {start|stop|restart|status|logs|enable|disable}"
}

start_service() {
    log_info "Запуск сервиса $SERVICE_API..."
    systemctl start "$SERVICE_API"
    log_success "Сервис запущен"
}

stop_service() {
    log_info "Остановка сервиса $SERVICE_API..."
    systemctl stop "$SERVICE_API"
    log_success "Сервис остановлен"
}

restart_service() {
    log_info "Перезапуск сервиса $SERVICE_API..."
    systemctl restart "$SERVICE_API"
    log_success "Сервис перезапущен"
}

status_service() {
    echo "=== Статус сервиса $SERVICE_API ==="
    systemctl status "$SERVICE_API" --no-pager -l
}

show_logs() {
    journalctl -u "$SERVICE_API" -f -n 50
}

enable_service() {
    log_info "Включение автозагрузки $SERVICE_API..."
    systemctl enable "$SERVICE_API"
    log_success "Автозагрузка включена"
}

disable_service() {
    log_info "Отключение автозагрузки $SERVICE_API..."
    systemctl disable "$SERVICE_API"
    log_success "Автозагрузка отключена"
}

case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        show_logs
        ;;
    enable)
        enable_service
        ;;
    disable)
        disable_service
        ;;
    *)
        show_usage
        exit 1
esac
EOF

    chmod +x "$SCRIPTS_DIR/manage.sh"
    log_success "Создан скрипт управления: $SCRIPTS_DIR/manage.sh"
    
    # Скрипт проверки статуса
    cat > "$SCRIPTS_DIR/status.sh" << 'EOF'
#!/bin/bash

echo "=== Статус BlackFox Services ==="
echo ""

# API сервис
echo "API Service (blackfox-api.service):"
if systemctl is-active blackfox-api.service &> /dev/null; then
    echo -e "   Status: \033[32mACTIVE\033[0m"
    echo -e "   Enabled: \033[33m$(systemctl is-enabled blackfox-api.service)\033[0m"
else
    echo -e "   Status: \033[31mINACTIVE\033[0m"
fi

# Nginx
echo ""
echo "Nginx:"
if systemctl is-active nginx &> /dev/null; then
    echo -e "   Status: \033[32mACTIVE\033[0m"
else
    echo -e "   Status: \033[33mNOT INSTALLED/INACTIVE\033[0m"
fi

# Проверка портов
echo ""
echo "Port Check:"
if netstat -tuln | grep -q ':8000'; then
    echo -e "   Port 8000 (API): \033[32mLISTENING\033[0m"
else
    echo -e "   Port 8000 (API): \033[31mNOT LISTENING\033[0m"
fi

if netstat -tuln | grep -q ':80'; then
    echo -e "   Port 80 (HTTP): \033[32mLISTENING\033[0m"
else
    echo -e "   Port 80 (HTTP): \033[33mNOT LISTENING\033[0m"
fi

echo ""
echo "Quick Commands:"
echo "  Start API:    systemctl start blackfox-api.service"
echo "  Stop API:     systemctl stop blackfox-api.service"
echo "  View Logs:    journalctl -u blackfox-api.service -f"
echo "  API Health:   curl http://localhost:8000/health"
EOF

    chmod +x "$SCRIPTS_DIR/status.sh"
    log_success "Создан скрипт статуса: $SCRIPTS_DIR/status.sh"
}

# Основное развертывание
deploy() {
    log_info "=== Начало развертывания BlackFox ==="
    
    check_privileges
    check_directories
    
    # Поиск virtualenv
    local venv_path=$(find_poetry_venv)
    
    # Создание systemd сервиса
    create_api_service "$venv_path"
    
    # Опциональная настройка Nginx
    setup_nginx_optional
    
    # Создание скриптов управления
    create_management_scripts
    
    # Перезагрузка systemd
    log_info "Перезагрузка systemd..."
    systemctl daemon-reload
    
    # Включение автозагрузки
    log_info "Включение автозагрузки сервиса..."
    systemctl enable "$PROJECT_NAME-api.service"
    
    log_success "=== Развертывание завершено! ==="
    echo ""
    echo "Следующие шаги:"
    echo "  1. Запустите сервис: $SCRIPTS_DIR/manage.sh start"
    echo "  2. Проверьте статус: $SCRIPTS_DIR/status.sh"
    echo "  3. Просмотрите логи: $SCRIPTS_DIR/manage.sh logs"
    echo ""
    echo "API будет доступно по: http://localhost:8000"
    echo "Health check: http://localhost:8000/health"
    echo "Metrics: http://localhost:8000/metrics"
    echo ""
    echo "Если нужен доступ извне, настройте фаервол:"
    echo "  ufw allow 8000/tcp"
}

# Запуск
deploy