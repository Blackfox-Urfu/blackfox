# Dockerfile

# 1. Используем официальный образ Python
FROM python:3.12-slim

# 2. Устанавливаем рабочую директорию
WORKDIR /app

# 3. Копируем файлы с зависимостями
COPY pyproject.toml poetry.lock ./

# 4. Устанавливаем Poetry и зависимости проекта
# 'virtualenvs.create false' устанавливает зависимости в /usr/local/ а не в вирт. окружение
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --only main

# 5. Копируем исходный код приложения
COPY . .

# 6. Указываем команду для запуска сервера
# Путь к вашему главному файлу main.py
CMD ["uvicorn", "app.server_run.main:app", "--host", "0.0.0.0", "--port", "8000"]