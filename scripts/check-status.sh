#!/bin/bash

echo "=== BlackFox Status Check ==="
echo ""

# Service status
echo "1. Service Status:"
systemctl is-active blackfox-api.service > /dev/null && echo -e "   ✅ blackfox-api.service: ACTIVE" || echo -e "   ❌ blackfox-api.service: INACTIVE"

# Port check
echo ""
echo "2. Port Check:"
netstat -tuln | grep -q ':8000' && echo -e "   ✅ Port 8000: LISTENING" || echo -e "   ❌ Port 8000: NOT LISTENING"

# API health
echo ""
echo "3. API Health:"
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "   ✅ Health endpoint: RESPONDING"
    echo "   Response:"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
else
    echo -e "   ❌ Health endpoint: NOT RESPONDING"
fi

# Nginx status
echo ""
echo "4. Nginx Status:"
if systemctl is-active nginx > /dev/null 2>&1; then
    echo -e "   ✅ Nginx: ACTIVE"
    # Check if our config is loaded
    if nginx -T 2>/dev/null | grep -q "blackfoxus.ru"; then
        echo -e "   ✅ Domain config: LOADED"
    else
        echo -e "   ⚠️ Domain config: NOT LOADED"
    fi
else
    echo -e "   ⚠️ Nginx: INACTIVE (optional)"
fi

# Model status from health endpoint
echo ""
echo "5. Model Status:"
curl -s http://localhost:8000/health | grep -o '"model_loaded":[^,]*' | sed 's/"/ /g' | sed 's/^/   /'

echo ""
echo "=== Access URLs ==="
echo "Local API:    http://localhost:8000"
echo "Health:       http://localhost:8000/health"
echo "Domain:       http://blackfoxus.ru"
echo "Domain Health: http://blackfoxus.ru/health"
