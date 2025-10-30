#!/bin/bash
set -e

echo "=== BlackFox Final Deploy ==="

# Поиск virtualenv
cd /root/blackfox
VENV_PATH=$(poetry env info --path)
echo "Virtualenv: $VENV_PATH"

# Создание systemd сервиса
cat > /etc/systemd/system/blackfox-api.service << EOL
[Unit]
Description=BlackFox API Service
After=network.target

[Service]
Type=exec
User=root
WorkingDirectory=/root/blackfox
Environment="PYTHONPATH=/root/blackfox"
ExecStart=$VENV_PATH/bin/uvicorn app.server_run.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload
systemctl enable blackfox-api.service

echo "Deployment complete!"
echo "Start: systemctl start blackfox-api.service"
echo "Status: systemctl status blackfox-api.service"
echo "Logs: journalctl -u blackfox-api.service -f"
