#!/bin/bash
# DeepSeek Environment Setup Script
# This script sets up the environment for the DeepSeek-R1-Distill-Qwen-7B reasoning component
# Usage: bash setup-deepseek.sh

# Exit on any error
set -e

echo "=== Starting DeepSeek environment setup ==="

# Detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
  OS_TYPE="macos"
  echo "Detected macOS operating system"
else
  OS_TYPE="linux"
  echo "Detected Linux-based operating system"
fi

# Ask user about domain and SSL
echo ""
echo "Do you want to set up SSL with Let's Encrypt? (y/n)"
read -p "This requires a domain name pointing to this server: " setup_ssl

if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
  read -p "Enter your domain name (e.g., api.example.com): " domain_name
  if [ -z "$domain_name" ]; then
    echo "No domain provided. Continuing without SSL setup."
    setup_ssl="n"
  fi
fi

# Update system packages and install dependencies based on OS
echo "Updating system packages..."
if [ "$OS_TYPE" = "macos" ]; then
  # macOS specific setup
  if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
  
  echo "Installing Python and development tools..."
  brew update
  
  # Install required build dependencies for Python packages
  echo "Installing build dependencies..."
  brew install python3 git cmake pkg-config coreutils
  
  # Install nginx if SSL setup is requested
  if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
    brew install nginx
  fi
else
  # Linux specific setup
  sudo apt update
  sudo apt upgrade -y
  
  echo "Installing Python and development tools..."
  sudo apt install -y python3 python3-pip python3-dev python3-venv build-essential git cmake pkg-config
  
  # Install nginx and certbot if SSL setup is requested
  if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
    sudo apt install -y nginx certbot python3-certbot-nginx
  fi
fi

# Create application directory
echo "Creating application directory..."
APP_DIR=~/deepseek-app
mkdir -p $APP_DIR
mkdir -p $APP_DIR/api
mkdir -p $APP_DIR/logs
cd $APP_DIR

# Set up Python virtual environment for both macOS and Linux
echo "Setting up Python virtual environment..."
python3 -m venv venv
if [ "$OS_TYPE" = "macos" ]; then
  source venv/bin/activate
  PYTHON_CMD="$APP_DIR/venv/bin/python3"
  PIP_CMD="$APP_DIR/venv/bin/pip3"
else
  source venv/bin/activate
  PYTHON_CMD="$APP_DIR/venv/bin/python3"
  PIP_CMD="$APP_DIR/venv/bin/pip3"
fi

# Upgrade pip and install dependencies
echo "Installing Python packages..."
$PIP_CMD install --upgrade pip

# For Mac, try to install a pre-built binary of sentencepiece first, then continue with remaining packages
if [ "$OS_TYPE" = "macos" ]; then
  echo "Installing pre-built sentencepiece package for macOS..."
  $PIP_CMD install --upgrade pip wheel setuptools
  
  # Try using binary package if possible
  $PIP_CMD install --no-build-isolation sentencepiece
fi

echo "Installing remaining Python packages..."
$PIP_CMD install torch fastapi uvicorn requests

# Install Hugging Face Transformers and related libraries
echo "Installing model dependencies..."
$PIP_CMD install transformers accelerate protobuf

# Create a model directory where we'll download the model
mkdir -p $APP_DIR/models

# Set up swap space only on Linux
if [ "$OS_TYPE" = "linux" ]; then
  echo "Setting up swap space for better memory management..."
  if [ ! -f /swapfile ]; then
    sudo fallocate -l 16G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap file created and enabled."
  else
    echo "Swap file already exists."
  fi
else
  echo "Skipping swap setup on macOS as it uses different memory management."
fi

# Set up SSL if requested
if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
  echo "Setting up Nginx..."
  
  if [ "$OS_TYPE" = "macos" ]; then
    # macOS Nginx configuration
    NGINX_CONF_DIR="/usr/local/etc/nginx/servers"
    mkdir -p $NGINX_CONF_DIR
    
    cat > /tmp/deepseek_nginx << EOF
server {
    listen 80;
    server_name $domain_name;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF
    
    sudo mv /tmp/deepseek_nginx $NGINX_CONF_DIR/deepseek.conf
    brew services restart nginx
    
    echo "NOTE: On macOS, Let's Encrypt automation is limited. Please consider:"
    echo "1. Use 'brew install certbot' and configure it manually, or"
    echo "2. Use a reverse proxy service like Cloudflare, or"
    echo "3. Get a certificate from a certificate authority and install it manually"
    
  else
    # Linux Nginx and Let's Encrypt configuration
    cat > /tmp/deepseek_nginx << EOF
server {
    server_name $domain_name;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

    sudo mv /tmp/deepseek_nginx /etc/nginx/sites-available/deepseek
    sudo ln -sf /etc/nginx/sites-available/deepseek /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl restart nginx
    
    # Get SSL certificate
    echo "Obtaining SSL certificate from Let's Encrypt..."
    sudo certbot --nginx -d $domain_name --non-interactive --agree-tos --email admin@$domain_name
  fi
  
  echo "SSL setup initiated for $domain_name"
fi

# Create service configuration based on OS
echo "Creating service configuration..."
if [ "$OS_TYPE" = "macos" ]; then
  # macOS launchd plist
  cat > ~/Library/LaunchAgents/com.deepseek.api.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.deepseek.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>$APP_DIR/venv/bin/python3</string>
        <string>$APP_DIR/api/app.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$APP_DIR/api</string>
    <key>StandardOutPath</key>
    <string>$APP_DIR/logs/deepseek.log</string>
    <key>StandardErrorPath</key>
    <string>$APP_DIR/logs/deepseek-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>$APP_DIR</string>
    </dict>
</dict>
</plist>
EOF
  
  echo "LaunchAgent created. It will be loaded in the install.sh script."
  
else
  # Linux systemd service
  cat > /tmp/deepseek.service << EOF
[Unit]
Description=DeepSeek R1 Reasoning API Service
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR/api
ExecStart=$APP_DIR/venv/bin/python3 app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="PYTHONPATH=$APP_DIR"

[Install]
WantedBy=multi-user.target
EOF

  sudo mv /tmp/deepseek.service /etc/systemd/system/deepseek.service
  sudo systemctl daemon-reload
  sudo systemctl enable deepseek
fi

# Deactivate the virtual environment at the end
deactivate || true

echo "=== Setup complete! ==="
echo "The environment has been prepared for DeepSeek-R1-Distill-Qwen-7B"
echo ""
echo "Next steps:"
echo "1. Copy your application code to $APP_DIR/api/app.py"

if [ "$OS_TYPE" = "macos" ]; then
  echo "2. Start the service with: launchctl load ~/Library/LaunchAgents/com.deepseek.api.plist"
  echo "3. Check the service status with: launchctl list | grep com.deepseek"
else
  echo "2. Start the service with: sudo systemctl start deepseek"
  echo "3. Check the service status with: sudo systemctl status deepseek"
fi

echo ""
if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
  echo "Your API will be available at: https://$domain_name/"
else
  if [ "$OS_TYPE" = "macos" ]; then
    echo "Your API will be available at: http://localhost:8000/"
  else
    echo "Your API will be available at: http://YOUR_SERVER_IP:8000/"
  fi
fi