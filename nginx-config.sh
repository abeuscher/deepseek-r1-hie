#!/bin/bash
# Configure Nginx with extended timeouts for DeepSeek API
# Usage: sudo bash nginx-config.sh

echo "=== Setting up Nginx with extended timeouts for DeepSeek API ==="

# Check if Nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "Nginx is not installed. Installing now..."
    apt update
    apt install -y nginx
fi

# Create Nginx configuration file with extended timeouts
cat > /etc/nginx/sites-available/deepseek << EOF
server {
    listen 80;
    server_name _;  # Replace with your domain if applicable
    
    # Very long timeouts (30 minutes = 1800 seconds)
    proxy_connect_timeout 1800s;
    proxy_send_timeout 1800s;
    proxy_read_timeout 1800s;
    send_timeout 1800s;
    
    # Increase buffer size for large responses
    proxy_buffer_size 16k;
    proxy_buffers 8 16k;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # These timeouts need to be set in the location block as well
        proxy_connect_timeout 1800s;
        proxy_send_timeout 1800s;
        proxy_read_timeout 1800s;
    }
}
EOF

# Enable the site
ln -sf /etc/nginx/sites-available/deepseek /etc/nginx/sites-enabled/

# Test Nginx configuration
nginx -t

# Reload Nginx to apply changes
systemctl reload nginx

echo "=== Nginx configured with extended timeouts (30 minutes) ==="
echo "Your API should now handle long-running requests without timing out."
```
