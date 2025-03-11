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

# Ensure cache directory exists
echo "Ensuring cache directory exists..."
mkdir -p ~/deepseek-app/cache

# Clear DeepSeek application cache
echo "Clearing application cache..."
rm -rf ~/deepseek-app/cache/*

# Update the code if needed
echo "Checking for code updates..."
if [ -d "$HOME/deepseek-r1-hie" ]; then
  cd $HOME/deepseek-r1-hie
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

# Wait for service to be fully up - much longer wait time (5 minutes)
echo "Waiting for service to initialize..."
echo "This may take 2-5 minutes while the DeepSeek R1 model loads into memory..."

# Show a progress indicator during the wait
WAIT_TIME=300  # 5 minutes in seconds
INTERVAL=15    # Update progress every 15 seconds
STEPS=$((WAIT_TIME / INTERVAL))

for i in $(seq 1 $STEPS); do
  # Calculate percentage and create progress bar
  PERCENT=$((i * 100 / STEPS))
  ELAPSED=$((i * INTERVAL))
  REMAINING=$((WAIT_TIME - ELAPSED))
  
  echo -ne "\r[$PERCENT%] Loading model... elapsed: ${ELAPSED}s, est. remaining: ${REMAINING}s"
  
  # Check if the service is responding early
  if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "\r[100%] Service initialized successfully! (took ${ELAPSED} seconds)                    "
    break
  fi
  
  sleep $INTERVAL
done

echo "" # New line after progress bar

# Check service status
echo "Checking service status..."
systemctl status deepseek

echo "=== Clean restart completed ==="
echo "The API should now be available with a fresh instance"

# Test health endpoint with retry logic
echo "Testing health endpoint..."
MAX_RETRIES=10
RETRY_INTERVAL=30  # 30 seconds between retries
retry_count=0
success=false

while [ $retry_count -lt $MAX_RETRIES ] && [ "$success" = false ]; do
  if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ API is responding correctly"
    success=true
  else
    retry_count=$((retry_count+1))
    if [ $retry_count -lt $MAX_RETRIES ]; then
      echo "API not ready yet, retrying in ${RETRY_INTERVAL} seconds (attempt $retry_count of $MAX_RETRIES)..."
      sleep $RETRY_INTERVAL
    else
      echo "❌ API failed to respond correctly after $MAX_RETRIES attempts"
    fi
  fi
done

# Check if Nginx is configured
if [ -f "/etc/nginx/sites-enabled/deepseek" ]; then
  echo "Reloading Nginx configuration..."
  systemctl reload nginx
  echo "✅ Nginx configuration reloaded"
fi
