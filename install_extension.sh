#!/bin/bash

# Определяем операционную систему
OS="$(uname -s)"
CHROME_CMD=""

# Определяем команду для запуска Chrome в зависимости от ОС
case "${OS}" in
    Linux*)
        if command -v google-chrome &> /dev/null; then
            CHROME_CMD="google-chrome"
        elif command -v chromium &> /dev/null; then
            CHROME_CMD="chromium"
        elif command -v chromium-browser &> /dev/null; then
            CHROME_CMD="chromium-browser"
        else
            echo "Chrome/Chromium не установлен в системе Linux"
            echo "Попробуйте установить chromium-browser:"
            echo "sudo apt install chromium-browser  # для Ubuntu/Debian"
            echo "sudo pacman -S chromium           # для Arch Linux"
            echo "sudo dnf install chromium         # для Fedora"
            exit 1
        fi
        ;;
    Darwin*)
        if [ -d "/Applications/Google Chrome.app" ]; then
            CHROME_CMD="/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
        else
            echo "Chrome не установлен в системе MacOS"
            exit 1
        fi
        ;;
    MINGW*|CYGWIN*|MSYS*)
        if [ -f "/c/Program Files/Google/Chrome/Application/chrome.exe" ]; then
            CHROME_CMD="/c/Program Files/Google/Chrome/Application/chrome.exe"
        elif [ -f "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" ]; then
            CHROME_CMD="/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
        else
            echo "Chrome не установлен в системе Windows"
            exit 1
        fi
        ;;
    *)
        echo "Неподдерживаемая операционная система: ${OS}"
        exit 1
        ;;
esac

# Путь к директории с расширением
EXTENSION_PATH="$(pwd)/chrome"

# Проверяем существование директории
if [ ! -d "$EXTENSION_PATH" ]; then
    echo "Директория расширения не найдена: $EXTENSION_PATH"
    exit 1
fi

# Устанавливаем расширение
echo "Попытка установки расширения..."
echo "Путь к расширению: ${EXTENSION_PATH}"

# Получаем абсолютный путь к директории расширения
ABSOLUTE_PATH=$(readlink -f "${EXTENSION_PATH}")

# Закрываем все существующие процессы Chrome/Chromium
echo "Закрываем существующие процессы браузера..."
pkill chrome
pkill chromium
sleep 2

# Запускаем браузер с расширением
echo "Запускаем браузер с расширением..."
eval "${CHROME_CMD}" \
    --enable-extensions \
    --load-extension="${ABSOLUTE_PATH}" \
    --no-first-run \
    --no-default-browser-check \
    --debug-extensions \
    --ignore-certificate-errors \
    --ignore-urlfetcher-cert-requests \
    --allow-insecure-localhost \
    --user-data-dir=/tmp/chrome-dev \
    chrome://extensions &

echo "Команда запуска: ${CHROME_CMD} с игнорированием ошибок сертификата"
echo ""
echo "ВНИМАНИЕ: Браузер запущен в небезопасном режиме для разработки."
echo "Не рекомендуется использовать эти настройки для обычной работы в интернете."
echo ""
echo "Если расширение не появилось автоматически:"
echo "1. Откройте chrome://extensions"
echo "2. Включите 'Режим разработчика' (переключатель в правом верхнем углу)"
echo "3. Убедитесь, что расширение появилось в списке"
echo "4. Перейдите на https://blackfoxus.ru:8000 и подтвердите исключение безопасности"
echo ""
echo "Путь к расширению: ${ABSOLUTE_PATH}" 