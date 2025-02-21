#!/bin/bash
# DeepSeek Environment Setup Script
# This script sets up the environment for the DeepSeek-R1-Distill-Qwen-7B reasoning component
# Usage: bash setup-deepseek.sh

# Exit on any error
set -e

echo "=== Starting DeepSeek environment setup ==="

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

# Update system packages
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install required packages
echo "Installing Python and development tools..."
sudo apt install -y python3 python3-pip python3-dev python3-venv build-essential git

# Create application directory
echo "Creating application directory..."
APP_DIR=~/deepseek-app
mkdir -p $APP_DIR
cd $APP_DIR

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch fastapi uvicorn requests 

# Install Hugging Face Transformers and related libraries
echo "Installing model dependencies..."
pip install transformers accelerate sentencepiece protobuf

# Create the API service directory
mkdir -p $APP_DIR/api
cd $APP_DIR/api

# Create a model directory where we'll download the model
mkdir -p $APP_DIR/models

# Create a swap file for better memory management
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

# Set up SSL if requested
if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
  echo "Setting up Nginx and Let's Encrypt..."
  sudo apt install -y nginx certbot python3-certbot-nginx
  
  # Configure Nginx
  echo "Configuring Nginx..."
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
  
  echo "SSL setup complete for $domain_name"
fi

# Create systemd service file
echo "Creating systemd service file..."
cat > /tmp/deepseek.service << EOF
[Unit]
Description=DeepSeek R1 Reasoning API Service
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR/api
ExecStart=$APP_DIR/venv/bin/python app.py
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

echo "=== Setup complete! ==="
echo "The environment has been prepared for DeepSeek-R1-Distill-Qwen-7B"
echo ""
echo "Next steps:"
echo "1. Copy your application code to $APP_DIR/api/app.py"
echo "2. Start the service with: sudo systemctl start deepseek"
echo "3. Check the service status with: sudo systemctl status deepseek"
echo ""
if [ "$setup_ssl" = "y" ] || [ "$setup_ssl" = "Y" ]; then
  echo "Your API will be available at: https://$domain_name/"
else
  echo "Your API will be available at: http://YOUR_SERVER_IP:8000/"
fi