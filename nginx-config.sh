#!/bin/bash
# Configure Nginx with extended timeouts for DeepSeek API while preserving SSL settings
# Usage: sudo bash nginx-config.sh

echo "=== Setting up Nginx with extended timeouts for DeepSeek API ==="

# Check if Nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "Nginx is not installed. Installing now..."
    apt update
    apt install -y nginx
fi

# Backup existing configuration if it exists
if [ -f /etc/nginx/sites-available/deepseek ]; then
    echo "Backing up existing Nginx configuration..."
    cp /etc/nginx/sites-available/deepseek /etc/nginx/sites-available/deepseek.bak.$(date +%Y%m%d%H%M%S)
fi

# Check for existing SSL certificate paths
SSL_CERT=""
SSL_KEY=""
DOMAIN_NAME=""

if [ -f /etc/nginx/sites-available/deepseek ]; then
    # Extract SSL certificate paths and domain name from existing config if they exist
    SSL_CERT=$(grep -oP "ssl_certificate \K[^;]+" /etc/nginx/sites-available/deepseek | tr -d ' ' || echo "")
    SSL_KEY=$(grep -oP "ssl_certificate_key \K[^;]+" /etc/nginx/sites-available/deepseek | tr -d ' ' || echo "")
    DOMAIN_NAME=$(grep -oP "server_name \K[^;]+" /etc/nginx/sites-available/deepseek | tr -d ' ' || echo "_")
    
    echo "Found existing configuration:"
    [ -n "$SSL_CERT" ] && echo "- SSL Certificate: $SSL_CERT"
    [ -n "$SSL_KEY" ] && echo "- SSL Key: $SSL_KEY"
    [ -n "$DOMAIN_NAME" ] && echo "- Domain: $DOMAIN_NAME"
fi

# Create Nginx configuration file with extended timeouts and preserved SSL settings if they exist
echo "Creating Nginx configuration with extended timeouts..."

if [ -n "$SSL_CERT" ] && [ -n "$SSL_KEY" ]; then
    # With SSL configuration
    cat > /etc/nginx/sites-available/deepseek << EOF
server {
    listen 80;
    server_name ${DOMAIN_NAME};
    
    # Redirect all HTTP requests to HTTPS
    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name ${DOMAIN_NAME};
    
    ssl_certificate ${SSL_CERT};
    ssl_certificate_key ${SSL_KEY};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    
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
else
    # Without SSL (HTTP only)
    cat > /etc/nginx/sites-available/deepseek << EOF
server {
    listen 80;
    server_name ${DOMAIN_NAME};
    
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
fi

# Enable the site (if not already enabled)
ln -sf /etc/nginx/sites-available/deepseek /etc/nginx/sites-enabled/

# Check and remove default site if it exists (to avoid conflicts)
if [ -f /etc/nginx/sites-enabled/default ]; then
    echo "Removing default Nginx site to avoid conflicts..."
    rm -f /etc/nginx/sites-enabled/default
fi

# Test Nginx configuration
echo "Testing Nginx configuration..."
nginx -t

# Reload Nginx to apply changes if configuration test passes
if [ $? -eq 0 ]; then
    echo "Configuration valid, reloading Nginx..."
    systemctl reload nginx
    echo "✅ Nginx configuration reloaded successfully"
else
    echo "❌ Nginx configuration test failed. Please check the configuration manually."
    echo "Restoring backup if available..."
    if [ -f /etc/nginx/sites-available/deepseek.bak.* ]; then
        LATEST_BACKUP=$(ls -t /etc/nginx/sites-available/deepseek.bak.* | head -1)
        cp $LATEST_BACKUP /etc/nginx/sites-available/deepseek
        systemctl reload nginx
        echo "Restored previous configuration from $LATEST_BACKUP"
    fi
fi

echo "=== Nginx configured with extended timeouts (30 minutes) ==="

# Display connection information
if [ -n "$DOMAIN_NAME" ] && [ "$DOMAIN_NAME" != "_" ]; then
    if [ -n "$SSL_CERT" ] && [ -n "$SSL_KEY" ]; then
        echo "Your API should now be available at: https://$DOMAIN_NAME/"
    else
        echo "Your API should now be available at: http://$DOMAIN_NAME/"
    fi
else
    IP_ADDR=$(hostname -I | awk '{print $1}')
    echo "Your API should now be available at: http://$IP_ADDR/"
fi

echo "You can check if the API is responding with: curl -k http://localhost:8000/health"
