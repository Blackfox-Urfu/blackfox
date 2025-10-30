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

PROJECT_NAME="blackfox"
PROJECT_ROOT="/root/$PROJECT_NAME"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Проверка и создание директории scripts
mkdir -p "$SCRIPTS_DIR"

# Копирование скриптов
log_info "Инициализация скриптов развертывания..."

if [[ -f "deploy.sh" ]]; then
    cp deploy.sh "$SCRIPTS_DIR/"
    chmod +x "$SCRIPTS_DIR/deploy.sh"
    log_success "Скрипт deploy.sh скопирован"
else
    log_error "Файл deploy.sh не найден в текущей директории"
    exit 1
fi

# Создание README для скриптов
cat > "$SCRIPTS_DIR/README.md" << EOF
# Скрипты развертывания BlackFox

## Структура

- \`deploy.sh\` - основной скрипт развертывания
- \`manage.sh\` - скрипт управления сервисами (создается автоматически)
- \`update.sh\` - скрипт обновления (создается автоматически)

## Использование

1. Инициализация (выполняется один раз):
   \`\`\`bash
   ./init.sh
   \`\`\`

2. Развертывание:
   \`\`\`bash
   $SCRIPTS_DIR/deploy.sh
   \`\`\`

3. Управление сервисами:
   \`\`\`bash
   $SCRIPTS_DIR/manage.sh start    # запуск
   $SCRIPTS_DIR/manage.sh stop     # остановка
   $SCRIPTS_DIR/manage.sh status   # статус
   $SCRIPTS_DIR/manage.sh logs     # логи
   \`\`\`

4. Обновление:
   \`\`\`bash
   $SCRIPTS_DIR/update.sh
   \`\`\`
EOF

log_success "Инициализация завершена!"
echo ""
echo "Для развертывания выполните: $SCRIPTS_DIR/deploy.sh"