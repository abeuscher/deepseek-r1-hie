#!/bin/bash
# DeepSeek R1 Uninstallation Script
# This script completely removes the DeepSeek-R1 installation

echo "=== Starting DeepSeek-R1 uninstallation ==="

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
  OS_TYPE="macos"
  echo "Detected macOS operating system"
else
  OS_TYPE="linux"
  echo "Detected Linux operating system"
fi

# Stop the service based on OS
if [ "$OS_TYPE" = "macos" ]; then
  # macOS service
  echo "Stopping and removing macOS service..."
  launchctl unload ~/Library/LaunchAgents/com.deepseek.api.plist 2>/dev/null || true
  rm ~/Library/LaunchAgents/com.deepseek.api.plist 2>/dev/null || true
else
  # Linux service
  echo "Stopping and removing Linux service..."
  sudo systemctl stop deepseek 2>/dev/null || true
  sudo systemctl disable deepseek 2>/dev/null || true
  sudo rm /etc/systemd/system/deepseek.service 2>/dev/null || true
  sudo systemctl daemon-reload
fi

# Remove application files
echo "Removing application directory..."
rm -rf ~/deepseek-app

# Clean up Nginx configuration if it exists
if [ "$OS_TYPE" = "macos" ]; then
  # macOS Nginx configuration
  if [ -f /usr/local/etc/nginx/servers/deepseek.conf ]; then
    echo "Removing Nginx configuration..."
    rm /usr/local/etc/nginx/servers/deepseek.conf
    brew services restart nginx 2>/dev/null || true
  fi
else
  # Linux Nginx configuration
  if [ -f /etc/nginx/sites-enabled/deepseek ]; then
    echo "Removing Nginx configuration..."
    sudo rm /etc/nginx/sites-enabled/deepseek
    sudo rm /etc/nginx/sites-available/deepseek 2>/dev/null || true
    sudo systemctl restart nginx 2>/dev/null || true
  fi
fi

# Remove swap file on Linux if we created one
if [ "$OS_TYPE" = "linux" ] && [ -f /swapfile ]; then
  echo "Do you want to remove the swap file that was created during installation? (y/n)"
  read -p "This is not recommended if you're using swap for other applications: " remove_swap
  
  if [ "$remove_swap" = "y" ] || [ "$remove_swap" = "Y" ]; then
    echo "Removing swap file..."
    sudo swapoff /swapfile
    sudo sed -i '/\/swapfile/d' /etc/fstab
    sudo rm /swapfile
  else
    echo "Keeping swap file as requested."
  fi
fi

echo "=== Uninstallation complete! ==="
echo "The DeepSeek-R1 service has been completely removed from your system."