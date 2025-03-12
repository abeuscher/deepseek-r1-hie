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
# Redirect output to log files instead of journal
StandardOutput=append:$HOME/deepseek-app/logs/deepseek.log
StandardError=append:$HOME/deepseek-app/logs/deepseek-error.log
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

# Configure Nginx with extended timeouts and SSL if applicable
setup_nginx() {
  echo "Setting up Nginx web server with extended timeouts..."
  
  # Check if Nginx is installed
  if ! command -v nginx &> /dev/null; then
    echo "Nginx is not installed. Installing now..."
    if [ "$PKG_MANAGER" = "apt" ]; then
      sudo apt update
      sudo apt install -y nginx
    elif [ "$PKG_MANAGER" = "dnf" ] || [ "$PKG_MANAGER" = "yum" ]; then
      sudo $PKG_MANAGER install -y nginx
    elif [ "$PKG_MANAGER" = "apk" ]; then
      sudo apk add nginx
    elif [ "$PKG_MANAGER" = "brew" ]; then
      brew install nginx
    else
      echo "Could not install Nginx. Please install manually and run this script again."
      return 1
    fi
  fi
  
  # Backup existing configuration if it exists
  if [[ "$OSTYPE" == "darwin"* ]]; then
    NGINX_CONF_DIR="/usr/local/etc/nginx/servers"
    NGINX_CONF_FILE="$NGINX_CONF_DIR/deepseek.conf"
    if [ -f "$NGINX_CONF_FILE" ]; then
      cp "$NGINX_CONF_FILE" "$NGINX_CONF_FILE.bak.$(date +%Y%m%d%H%M%S)"
    fi
  else
    NGINX_CONF_DIR="/etc/nginx/sites-available"
    NGINX_CONF_FILE="$NGINX_CONF_DIR/deepseek"
    if [ -f "$NGINX_CONF_FILE" ]; then
      sudo cp "$NGINX_CONF_FILE" "$NGINX_CONF_FILE.bak.$(date +%Y%m%d%H%M%S)"
    fi
  fi
  
  # Check for existing SSL certificate paths and domain
  SSL_CERT=""
  SSL_KEY=""
  DOMAIN_NAME=""
  
  if [ -f "$NGINX_CONF_FILE" ]; then
    # Extract settings from existing config if they exist
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # Fix: Better parsing of SSL certificate paths
      SSL_CERT=$(grep -oP "ssl_certificate\s+\K[^;]+" "$NGINX_CONF_FILE" 2>/dev/null | tr -d ' ' || echo "")
      SSL_KEY=$(grep -oP "ssl_certificate_key\s+\K[^;]+" "$NGINX_CONF_FILE" 2>/dev/null | tr -d ' ' || echo "")
      DOMAIN_NAME=$(grep -oP "server_name\s+\K[^;]+" "$NGINX_CONF_FILE" 2>/dev/null | awk '{print $1}' || echo "_")
    else
      # Fix: Better parsing of SSL certificate paths
      SSL_CERT=$(sudo grep -oP "ssl_certificate\s+\K[^;]+" "$NGINX_CONF_FILE" 2>/dev/null | tr -d ' ' || echo "")
      SSL_KEY=$(sudo grep -oP "ssl_certificate_key\s+\K[^;]+" "$NGINX_CONF_FILE" 2>/dev/null | tr -d ' ' || echo "")
      DOMAIN_NAME=$(sudo grep -oP "server_name\s+\K[^;]+" "$NGINX_CONF_FILE" 2>/dev/null | awk '{print $1}' || echo "_")
    fi
    
    echo "Found existing Nginx configuration:"
    [ -n "$SSL_CERT" ] && [ "$SSL_CERT" != " " ] && echo "- SSL Certificate: $SSL_CERT"
    [ -n "$SSL_KEY" ] && [ "$SSL_KEY" != " " ] && echo "- SSL Key: $SSL_KEY"
    [ -n "$DOMAIN_NAME" ] && [ "$DOMAIN_NAME" != " " ] && echo "- Domain: $DOMAIN_NAME"
  else
    # No existing config, use hostname or IP for domain
    if [[ "$OSTYPE" == "darwin"* ]]; then
      DOMAIN_NAME="localhost"
    else
      DOMAIN_NAME=$(hostname -I | awk '{print $1}')
    fi
    echo "No existing Nginx configuration. Using domain: $DOMAIN_NAME"
  fi
  
  # Create Nginx configuration with extended timeouts
  if [[ -n "$SSL_CERT" && -n "$SSL_KEY" && "$SSL_CERT" != " " && "$SSL_KEY" != " " ]]; then
    # SSL configuration
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS Nginx config
      cat > /tmp/deepseek.conf << 'EOF'
server {
    listen 80;
    server_name ${DOMAIN_NAME};
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name ${DOMAIN_NAME};
    
    ssl_certificate ${SSL_CERT};
    ssl_certificate_key ${SSL_KEY};
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Very long timeouts (30 minutes = 1800 seconds)
    proxy_connect_timeout 1800s;
    proxy_send_timeout 1800s;
    proxy_read_timeout 1800s;
    send_timeout 1800s;
    
    # Increase buffer sizes
    proxy_buffer_size 16k;
    proxy_buffers 8 16k;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Location-specific timeout settings
        proxy_connect_timeout 1800s;
        proxy_send_timeout 1800s;
        proxy_read_timeout 1800s;
    }
}
EOF
      # Replace variables in the template
      sed -i '' "s/\${DOMAIN_NAME}/$DOMAIN_NAME/g; s/\${SSL_CERT}/$SSL_CERT/g; s/\${SSL_KEY}/$SSL_KEY/g" /tmp/deepseek.conf
      cp /tmp/deepseek.conf "$NGINX_CONF_FILE"
      rm /tmp/deepseek.conf
      brew services restart nginx
    else
      # Linux Nginx config - Fix: Using a different approach with sed
      cat > /tmp/deepseek << 'EOF'
server {
    listen 80;
    server_name SERVER_NAME_PLACEHOLDER;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name SERVER_NAME_PLACEHOLDER;
    
    ssl_certificate SSL_CERT_PLACEHOLDER;
    ssl_certificate_key SSL_KEY_PLACEHOLDER;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Very long timeouts (30 minutes = 1800 seconds)
    proxy_connect_timeout 1800;
    proxy_send_timeout 1800;
    proxy_read_timeout 1800;
    send_timeout 1800;
    
    # Increase buffer sizes
    proxy_buffer_size 16k;
    proxy_buffers 8 16k;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Location-specific timeout settings
        proxy_connect_timeout 1800;
        proxy_send_timeout 1800;
        proxy_read_timeout 1800;
    }
}
EOF
      # Replace placeholders with actual values
      sudo sed -i "s/SERVER_NAME_PLACEHOLDER/$DOMAIN_NAME/g; s|SSL_CERT_PLACEHOLDER|$SSL_CERT|g; s|SSL_KEY_PLACEHOLDER|$SSL_KEY|g" /tmp/deepseek
      sudo mv /tmp/deepseek "$NGINX_CONF_FILE"
      sudo ln -sf "$NGINX_CONF_FILE" /etc/nginx/sites-enabled/deepseek
      
      # Remove default site to avoid conflicts
      if [ -f /etc/nginx/sites-enabled/default ]; then
        sudo rm /etc/nginx/sites-enabled/default
      fi
      
      sudo systemctl reload nginx
    fi
  else
    # HTTP-only configuration - Apply similar fixes here
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS Nginx config
      cat > /tmp/deepseek.conf << 'EOF'
server {
    listen 80;
    server_name ${DOMAIN_NAME};
    
    # Very long timeouts (30 minutes = 1800 seconds)
    proxy_connect_timeout 1800s;
    proxy_send_timeout 1800s;
    proxy_read_timeout 1800s;
    send_timeout 1800s;
    
    # Increase buffer sizes
    proxy_buffer_size 16k;
    proxy_buffers 8 16k;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Location-specific timeout settings
        proxy_connect_timeout 1800s;
        proxy_send_timeout 1800s;
        proxy_read_timeout 1800s;
    }
}
EOF
      sed -i '' "s/\${DOMAIN_NAME}/$DOMAIN_NAME/g" /tmp/deepseek.conf
      cp /tmp/deepseek.conf "$NGINX_CONF_FILE"
      rm /tmp/deepseek.conf
      brew services restart nginx
    else
      # Linux Nginx config
      cat > /tmp/deepseek << 'EOF'
server {
    listen 80;
    server_name SERVER_NAME_PLACEHOLDER;
    
    # Very long timeouts (30 minutes = 1800 seconds)
    proxy_connect_timeout 1800;
    proxy_send_timeout 1800;
    proxy_read_timeout 1800;
    send_timeout 1800;
    
    # Increase buffer sizes
    proxy_buffer_size 16k;
    proxy_buffers 8 16k;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Location-specific timeout settings
        proxy_connect_timeout 1800;
        proxy_send_timeout 1800;
        proxy_read_timeout 1800;
    }
}
EOF
      sudo sed -i "s/SERVER_NAME_PLACEHOLDER/$DOMAIN_NAME/g" /tmp/deepseek
      sudo mv /tmp/deepseek "$NGINX_CONF_FILE"
      sudo ln -sf "$NGINX_CONF_FILE" /etc/nginx/sites-enabled/deepseek
      
      # Remove default site to avoid conflicts
      if [ -f /etc/nginx/sites-enabled/default ]; then
        sudo rm /etc/nginx/sites-enabled/default
      fi
      
      sudo systemctl reload nginx
    fi
  fi
  
  # Add testing of configuration before reloading
  echo "Testing Nginx configuration..."
  if ! sudo nginx -t; then
    echo "Nginx configuration failed validation. Reverting changes..."
    if [ -f "${NGINX_CONF_FILE}.bak.$(date +%Y%m%d%H%M%S)" ]; then
      sudo cp "${NGINX_CONF_FILE}.bak.$(date +%Y%m%d%H%M%S}" "$NGINX_CONF_FILE"
      echo "Restored previous configuration."
    fi
    return 1
  fi
  
  echo "Nginx configured with extended timeouts 30 minutes"
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

# Create modules directory
echo "Setting up modular structure..."
mkdir -p ~/deepseek-app/api/modules

# Copy module files
echo "Installing module files..."
cp modules/*.py ~/deepseek-app/api/modules/

# Install additional dependency
echo "Installing additional dependencies..."
~/deepseek-app/venv/bin/pip install psutil

# Start the service
echo "Starting the DeepSeek-R1 service..."
setup_service

# Configure Nginx with extended timeouts
echo "Configuring Nginx web server..."
setup_nginx

# Display status
echo "=== Installation complete! ==="

# Check service status
if [[ "$OSTYPE" == "darwin"* ]]; then
  launchctl list | grep com.deepseek.api
else
  sudo systemctl status deepseek
fi

# Check if Nginx was set up with a domain
if [[ "$OSTYPE" == "darwin"* ]]; then
  if [ -f "$NGINX_CONF_DIR/deepseek.conf" ]; then
    domain=$(grep server_name "$NGINX_CONF_DIR/deepseek.conf" | awk '{print $2}' | sed 's/;//')
    if grep -q "listen 443" "$NGINX_CONF_DIR/deepseek.conf"; then
      echo ""
      echo "Your API is now available at: https://$domain/"
    else
      echo ""
      echo "Your API is now available at: http://$domain/"
    fi
  else
    echo ""
    echo "Your API is now available at: http://localhost:8000/"
  fi
else
  if [ -f "/etc/nginx/sites-enabled/deepseek" ]; then
    domain=$(sudo grep server_name /etc/nginx/sites-available/deepseek | awk '{print $2}' | sed 's/;//' | head -1)
    if sudo grep -q "listen 443" /etc/nginx/sites-available/deepseek; then
      echo ""
      echo "Your API is now available at: https://$domain/"
    else
      echo ""
      echo "Your API is now available at: http://$domain/"
    fi
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