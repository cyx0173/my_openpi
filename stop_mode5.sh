#!/bin/bash
# Stop all Mode 5 servers (ports 8000-8003)

pkill -f "python scripts/serve_policy.py"
sleep 1

# Verify ports are free
for PORT in 8000 8001 8002 8003; do
    if PID=$(lsof -ti :$PORT 2>/dev/null); then
        echo "Warning: port $PORT still occupied by PID $PID"
    else
        echo "Port $PORT is free"
    fi
done

echo "Done."
