#!/bin/bash
# DeepSeek-R1 Installation Script
# This script sets up the complete DeepSeek-R1 API service
# Usage: bash install.sh

# Exit on any error
set -e

echo "=== Starting DeepSeek-R1 installation ==="

# Run the environment setup script
chmod +x setup-deepseek.sh
./setup-deepseek.sh

# Copy the app.py file to the installation directory
echo "Installing the API service..."
cp app.py ~/deepseek-app/api/

# Start the service
echo "Starting the DeepSeek-R1 service..."
sudo systemctl start deepseek

# Display status
echo "=== Installation complete! ==="
sudo systemctl status deepseek

# Check if Nginx is installed (indicating SSL was set up)
if [ -f /etc/nginx/sites-enabled/deepseek ]; then
  domain=$(grep server_name /etc/nginx/sites-available/deepseek | awk '{print $2}' | sed 's/;//')
  echo ""
  echo "Your API is now available at: https://$domain/"
else
  ip=$(hostname -I | awk '{print $1}')
  echo ""
  echo "Your API is now available at: http://$ip:8000/"
fi

echo ""
echo "API endpoints:"
echo "- /health (GET): Check if the service is running"
echo "- /process-context (POST): Process patient data and extract relevant context"
echo ""
echo "For more information, refer to the README.md file."