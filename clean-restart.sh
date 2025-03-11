#!/bin/bash
# Clean restart script for DeepSeek R1 service
# This script performs a complete cleanup and restart of the service
# Usage: sudo bash clean-restart.sh

echo "=== Starting DeepSeek-R1 clean restart process ==="

# First, stop the service
echo "Stopping DeepSeek service..."
systemctl stop deepseek

# Kill any remaining Python processes that might be related to DeepSeek
echo "Ensuring all related processes are terminated..."
pkill -f "python.*app.py" || echo "No lingering processes found"

# Make sure port 8000 is free
PORT_PROC=$(netstat -tulpn 2>/dev/null | grep ":8000" | awk '{print $7}' | cut -d'/' -f1)
if [ ! -z "$PORT_PROC" ]; then
  echo "Process $PORT_PROC is still using port 8000. Terminating..."
  kill -9 $PORT_PROC || echo "Failed to kill process, may need manual intervention"
fi

# Clear system cache (drops unused cache, not user data)
echo "Freeing system cache..."
sync
echo 3 > /proc/sys/vm/drop_caches

# Clear DeepSeek application cache
echo "Clearing application cache..."
rm -rf ~/deepseek-app/cache/*

# Update the code if needed
echo "Checking for code updates..."
if [ -d "~/deepseek-r1-hie" ]; then
  cd ~/deepseek-r1-hie
  if [ -d ".git" ]; then
    git pull
    # Copy updated files to the deployment directory
    cp app.py ~/deepseek-app/api/
    cp -r modules/* ~/deepseek-app/api/modules/
  fi
fi

# Restart the service
echo "Starting DeepSeek service..."
systemctl start deepseek

# Wait for service to be fully up
echo "Waiting for service to initialize..."
sleep 5

# Check service status
echo "Checking service status..."
systemctl status deepseek

echo "=== Clean restart completed ==="
echo "The API should now be available with a fresh instance"

# Test health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8000/health | grep -q "healthy" && \
  echo "✅ API is responding correctly" || \
  echo "❌ API failed to respond correctly"
```
