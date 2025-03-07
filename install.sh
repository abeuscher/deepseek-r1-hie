#!/bin/bash
# DeepSeek-R1 Installation Script
# This script sets up the complete DeepSeek-R1 API service
# Usage: bash install.sh
# Exit on any error
set -e

echo "=== Starting DeepSeek-R1 installation ==="

# Detect OS and package manager
detect_package_manager() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS detection
    echo "Detected macOS operating system"
    if command -v brew >/dev/null 2>&1; then
      echo "Homebrew is installed"
      PKG_MANAGER="brew"
      INSTALL_CMD="brew install"
      SERVICE_CMD="brew services"
    else
      echo "Homebrew is not installed. Installing Homebrew..."
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      PKG_MANAGER="brew"
      INSTALL_CMD="brew install"
      SERVICE_CMD="brew services"
    fi
  elif command -v apt >/dev/null 2>&1; then
    # Debian/Ubuntu
    echo "Detected Debian/Ubuntu-based system"
    PKG_MANAGER="apt"
    INSTALL_CMD="sudo apt install -y"
    SERVICE_CMD="sudo systemctl"
  elif command -v dnf >/dev/null 2>&1; then
    # Fedora/RHEL 8+
    echo "Detected Fedora/RHEL-based system"
    PKG_MANAGER="dnf"
    INSTALL_CMD="sudo dnf install -y"
    SERVICE_CMD="sudo systemctl"
  elif command -v yum >/dev/null 2>&1; then
    # CentOS/RHEL 7
    echo "Detected CentOS/RHEL-based system"
    PKG_MANAGER="yum"
    INSTALL_CMD="sudo yum install -y"
    SERVICE_CMD="sudo systemctl"
  elif command -v apk >/dev/null 2>&1; then
    # Alpine
    echo "Detected Alpine Linux"
    PKG_MANAGER="apk"
    INSTALL_CMD="sudo apk add"
    SERVICE_CMD="sudo rc-service"
  else
    echo "Could not detect package manager. Please install dependencies manually."
    exit 1
  fi
}

# Create service based on OS
setup_service() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS uses launchd instead of systemd
    echo "Setting up service for macOS..."
    
    # Create log directory
    mkdir -p ~/deepseek-app/logs
    
    cat > ~/Library/LaunchAgents/com.deepseek.api.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.deepseek.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>$HOME/deepseek-app/venv/bin/python3</string>
        <string>$HOME/deepseek-app/api/app.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$HOME/deepseek-app/api</string>
    <key>StandardOutPath</key>
    <string>$HOME/deepseek-app/logs/deepseek.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/deepseek-app/logs/deepseek-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>$HOME/deepseek-app</string>
    </dict>
</dict>
</plist>
EOF
    # Unload the service first if it exists to avoid errors
    launchctl unload ~/Library/LaunchAgents/com.deepseek.api.plist 2>/dev/null || true
    launchctl load ~/Library/LaunchAgents/com.deepseek.api.plist
  else
    # Linux systems with systemd
    echo "Setting up service for Linux..."
    cat > /tmp/deepseek.service << EOF
[Unit]
Description=DeepSeek R1 API Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$HOME/deepseek-app/api
ExecStart=$HOME/deepseek-app/venv/bin/python3 $HOME/deepseek-app/api/app.py
Restart=always
StandardOutput=journal
StandardError=journal
Environment="PYTHONPATH=$HOME/deepseek-app"

[Install]
WantedBy=multi-user.target
EOF
    sudo mv /tmp/deepseek.service /etc/systemd/system/deepseek.service
    sudo systemctl daemon-reload
    sudo systemctl enable deepseek
    sudo systemctl start deepseek
  fi
}

# Detect package manager
detect_package_manager

# Run the environment setup script
chmod +x setup-deepseek.sh
./setup-deepseek.sh

# Create deepseek-app directory structure if it doesn't exist
mkdir -p ~/deepseek-app/api
mkdir -p ~/deepseek-app/logs

# Copy the app.py file to the installation directory
echo "Installing the API service..."
cp app.py ~/deepseek-app/api/

# Start the service
echo "Starting the DeepSeek-R1 service..."
setup_service

# Display status
echo "=== Installation complete! ==="

# Check service status
if [[ "$OSTYPE" == "darwin"* ]]; then
  launchctl list | grep com.deepseek.api
else
  sudo systemctl status deepseek
fi

# Check if Nginx is installed (indicating SSL was set up)
if [[ "$OSTYPE" == "darwin"* ]]; then
  if [ -f /usr/local/etc/nginx/servers/deepseek.conf ]; then
    domain=$(grep server_name /usr/local/etc/nginx/servers/deepseek.conf | awk '{print $2}' | sed 's/;//')
    echo ""
    echo "Your API is now available at: https://$domain/"
  else
    echo ""
    echo "Your API is now available at: http://localhost:8000/"
  fi
else
  if [ -f /etc/nginx/sites-enabled/deepseek ]; then
    domain=$(grep server_name /etc/nginx/sites-available/deepseek | awk '{print $2}' | sed 's/;//')
    echo ""
    echo "Your API is now available at: https://$domain/"
  else
    ip=$(hostname -I | awk '{print $1}')
    echo ""
    echo "Your API is now available at: http://$ip:8000/"
  fi
fi

echo ""
echo "API endpoints:"
echo "- /health (GET): Check if the service is running"
echo "- /process-context (POST): Process patient data and extract relevant context"
echo ""
echo "For more information, refer to the README.md file."
echo ""
echo "IMPORTANT: To manually run the application, you must use the virtual environment:"
echo "  source ~/deepseek-app/venv/bin/activate"
echo "  python3 ~/deepseek-app/api/app.py"
echo "  # OR without activating the environment:"
echo "  ~/deepseek-app/venv/bin/python3 ~/deepseek-app/api/app.py"