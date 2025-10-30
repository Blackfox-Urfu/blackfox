#!/bin/bash

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Переменные
PROJECT_NAME="blackfox"
PROJECT_ROOT="/root/$PROJECT_NAME"
APP_DIR="$PROJECT_ROOT/app"
SERVER_RUN_DIR="$APP_DIR/server_run"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
SITE_DIR="$SERVER_RUN_DIR/site"
SERVICE_USER="root"
DOMAIN="blackfoxus.ru"
SSL_KEYFILE="/etc/letsencrypt/live/$DOMAIN/privkey.pem"
SSL_CERTFILE="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"

# Проверка прав
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Этот скрипт должен запускаться с правами root"
        exit 1
    fi
}

# Проверка существования директорий
check_directories() {
    log_info "Проверка структуры директорий..."
    
    local required_dirs=(
        "$PROJECT_ROOT"
        "$APP_DIR"
        "$SERVER_RUN_DIR"
        "$SCRIPTS_DIR"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_error "Директория $dir не существует"
            exit 1
        fi
    done
    
    log_success "Структура директорий проверена"
}

# Поиск virtualenv Poetry
find_poetry_venv() {
    log_info "Поиск virtualenv Poetry..."
    
    # Поиск в стандартных местах
    local possible_paths=(
        "/root/.cache/pypoetry/virtualenvs/$PROJECT_NAME-*"
        "/root/.cache/pypoetry/virtualenvs/blackfox-*"
        "$PROJECT_ROOT/.venv"
    )
    
    for path in "${possible_paths[@]}"; do
        if [[ -d $path ]]; then
            local found_path=$(ls -d $path 2>/dev/null | head -n1)
            if [[ -n "$found_path" && -f "$found_path/bin/activate" ]]; then
                echo "$found_path"
                log_success "Найден virtualenv: $found_path"
                return 0
            fi
        fi
    done
    
    log_error "Virtualenv Poetry не найден"
    log_info "Попробуйте выполнить: cd $PROJECT_ROOT && poetry install"
    exit 1
}

# Создание systemd сервиса для API
create_api_service() {
    local venv_path="$1"
    local service_file="/etc/systemd/system/$PROJECT_NAME-api.service"
    
    log_info "Создание systemd сервиса для API..."
    
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

ExecStart=$venv_path/bin/uvicorn app.server_run.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --ssl-keyfile $SSL_KEYFILE \
    --ssl-certfile $SSL_CERTFILE

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

# Создание systemd сервиса для сайта (Nginx)
create_site_service() {
    local service_file="/etc/systemd/system/$PROJECT_NAME-site.service"
    
    log_info "Создание systemd сервиса для сайта..."
    
    cat > "$service_file" << EOF
[Unit]
Description=BlackFox Static Site
After=network.target
Wants=network.target

[Service]
Type=exec
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$SITE_DIR

# Запуск простого HTTP сервера для статики (резервный вариант)
ExecStart=/usr/bin/python3 -m http.server 8080 --directory $SITE_DIR

# Или используем уже установленный nginx (предпочтительно)
# ExecStartPre=/usr/sbin/nginx -t
# ExecStart=/usr/sbin/nginx -g 'daemon off;'
# ExecReload=/usr/sbin/nginx -s reload

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    log_success "Создан systemd сервис для сайта: $service_file"
}

# Создание Nginx конфигурации
setup_nginx() {
    local nginx_dir="/etc/nginx/sites-available"
    local nginx_enabled_dir="/etc/nginx/sites-enabled"
    local config_file="$nginx_dir/$PROJECT_NAME"
    
    log_info "Настройка Nginx..."
    
    # Проверяем установлен ли nginx
    if ! command -v nginx &> /dev/null; then
        log_warning "Nginx не установлен. Установка..."
        apt update && apt install -y nginx
    fi
    
    cat > "$config_file" << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;

    ssl_certificate $SSL_CERTFILE;
    ssl_certificate_key $SSL_KEYFILE;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Статический сайт
    location / {
        root $SITE_DIR;
        index index.html;
        try_files \$uri \$uri/ =404;
        
        # Кэширование статических файлов
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # API прокси
    location /api/ {
        proxy_pass https://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Таймауты для ML-инференса
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health checks
    location /health {
        proxy_pass https://localhost:8000/health;
        proxy_set_header Host \$host;
        access_log off;
    }

    # Метрики Prometheus
    location /metrics {
        proxy_pass https://localhost:8000/metrics;
        proxy_set_header Host \$host;
        access_log off;
    }

    # Безопасность
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
}
EOF

    # Активация конфига nginx
    ln -sf "$config_file" "$nginx_enabled_dir/$PROJECT_NAME"
    
    # Проверка конфигурации
    if nginx -t; then
        log_success "Nginx конфигурация проверена"
    else
        log_error "Ошибка в конфигурации Nginx"
        exit 1
    fi
}

# Создание скрипта управления
create_management_script() {
    local script_path="$SCRIPTS_DIR/manage.sh"
    
    log_info "Создание скрипта управления..."
    
    cat > "$script_path" << 'EOF'
#!/bin/bash

set -e

PROJECT_NAME="blackfox"
SERVICE_API="$PROJECT_NAME-api.service"
SERVICE_SITE="$PROJECT_NAME-site.service"
NGINX_SERVICE="nginx"

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

show_usage() {
    echo "Использование: $0 {start|stop|restart|status|reload|logs|enable|disable}"
    echo ""
    echo "Команды:"
    echo "  start    - Запуск всех сервисов"
    echo "  stop     - Остановка всех сервисов"
    echo "  restart  - Перезапуск всех сервисов"
    echo "  status   - Статус сервисов"
    echo "  reload   - Перезагрузка конфигурации"
    echo "  logs     - Просмотр логов"
    echo "  enable   - Включение автозагрузки"
    echo "  disable  - Отключение автозагрузки"
}

start_services() {
    log_info "Запуск сервисов..."
    systemctl start "$SERVICE_API"
    systemctl start "$NGINX_SERVICE"
    log_success "Сервисы запущены"
}

stop_services() {
    log_info "Остановка сервисов..."
    systemctl stop "$SERVICE_API"
    systemctl stop "$NGINX_SERVICE"
    log_success "Сервисы остановлены"
}

restart_services() {
    log_info "Перезапуск сервисов..."
    systemctl restart "$SERVICE_API"
    systemctl restart "$NGINX_SERVICE"
    log_success "Сервисы перезапущены"
}

status_services() {
    echo "=== Статус сервисов ==="
    systemctl status "$SERVICE_API" --no-pager -l
    echo ""
    systemctl status "$NGINX_SERVICE" --no-pager -l
}

reload_services() {
    log_info "Перезагрузка конфигурации..."
    systemctl daemon-reload
    systemctl reload "$NGINX_SERVICE"
    log_success "Конфигурация перезагружена"
}

show_logs() {
    journalctl -u "$SERVICE_API" -f -n 50
}

enable_services() {
    log_info "Включение автозагрузки..."
    systemctl enable "$SERVICE_API"
    systemctl enable "$NGINX_SERVICE"
    log_success "Автозагрузка включена"
}

disable_services() {
    log_info "Отключение автозагрузки..."
    systemctl disable "$SERVICE_API"
    systemctl disable "$NGINX_SERVICE"
    log_success "Автозагрузка отключена"
}

case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        status_services
        ;;
    reload)
        reload_services
        ;;
    logs)
        show_logs
        ;;
    enable)
        enable_services
        ;;
    disable)
        disable_services
        ;;
    *)
        show_usage
        exit 1
esac

exit 0
EOF

    chmod +x "$script_path"
    log_success "Создан скрипт управления: $script_path"
}

# Создание скрипта обновления
create_update_script() {
    local script_path="$SCRIPTS_DIR/update.sh"
    
    log_info "Создание скрипта обновления..."
    
    cat > "$script_path" << 'EOF'
#!/bin/bash

set -e

PROJECT_NAME="blackfox"
PROJECT_ROOT="/root/$PROJECT_NAME"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

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

# Остановка сервисов перед обновлением
log_info "Остановка сервисов перед обновлением..."
"$SCRIPTS_DIR/manage.sh" stop

# Обновление кода из git (если используется)
if [[ -d "$PROJECT_ROOT/.git" ]]; then
    log_info "Обновление кода из git..."
    cd "$PROJECT_ROOT"
    git pull origin main
else
    log_warning "Git не инициализирован, пропускаем обновление кода"
fi

# Обновление зависимостей Poetry
log_info "Обновление зависимостей Poetry..."
cd "$PROJECT_ROOT"
poetry install --no-dev

# Перезапуск сервисов
log_info "Запуск сервисов после обновления..."
"$SCRIPTS_DIR/manage.sh" start

log_success "Обновление завершено успешно"
EOF

    chmod +x "$script_path"
    log_success "Создан скрипт обновления: $script_path"
}

# Основная функция развертывания
deploy() {
    log_info "Начало развертывания BlackFox..."
    
    check_privileges
    check_directories
    
    # Поиск virtualenv
    local venv_path=$(find_poetry_venv)
    
    # Создание сервисов
    create_api_service "$venv_path"
    create_site_service
    
    # Настройка Nginx
    setup_nginx
    
    # Создание скриптов управления
    create_management_script
    create_update_script
    
    # Перезагрузка systemd
    log_info "Перезагрузка systemd..."
    systemctl daemon-reload
    
    # Включение автозагрузки
    log_info "Включение автозагрузки сервисов..."
    systemctl enable "$PROJECT_NAME-api.service"
    systemctl enable nginx
    
    log_success "Развертывание завершено!"
    
    # Вывод следующх шагов
    echo ""
    echo "=== СЛЕДУЮЩИЕ ШАГИ ==="
    echo "1. Запустите сервисы: $SCRIPTS_DIR/manage.sh start"
    echo "2. Проверьте статус: $SCRIPTS_DIR/manage.sh status"
    echo "3. Просмотрите логи: $SCRIPTS_DIR/manage.sh logs"
    echo "4. Для обновления используйте: $SCRIPTS_DIR/update.sh"
    echo ""
    echo "API будет доступно по: https://$DOMAIN/api/"
    echo "Сайт будет доступен по: https://$DOMAIN/"
    echo "Метрики: https://$DOMAIN/metrics"
    echo "Health check: https://$DOMAIN/health"
}

# Обработка аргументов командной строки
case "${1:-}" in
    "--help" | "-h")
        echo "Использование: $0"
        echo ""
        echo "Автоматическое развертывание BlackFox ML приложения"
        echo ""
        echo "Требования:"
        echo "  - Установленный Poetry"
        echo "  - Существующий virtualenv Poetry"
        echo "  - SSL сертификаты в $SSL_CERTFILE"
        echo "  - Структура директорий проекта в $PROJECT_ROOT"
        ;;
    *)
        deploy
        ;;
esac