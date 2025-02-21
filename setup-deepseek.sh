#!/bin/bash
# DeepSeek Environment Setup Script
# This script sets up the environment for running DeepSeek R1 reasoning component
# Usage: bash setup-deepseek.sh

# Exit on any error
set -e

echo "=== Starting DeepSeek environment setup ==="

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

# Upgrade pip and install basic packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch fastapi uvicorn requests 

# Install additional packages for DeepSeek
echo "Installing DeepSeek dependencies..."
pip install accelerate transformers

# Clone DeepSeek-R1 repository
echo "Cloning DeepSeek-R1 repository..."
git clone https://github.com/deepseek-ai/DeepSeek-R1.git
cd DeepSeek-R1

# Install DeepSeek-R1 specific dependencies
echo "Installing DeepSeek-R1 specific dependencies..."
pip install -e .

echo "=== Setup complete! ==="
echo "To activate the environment:"
echo "  cd ~/deepseek-app"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Configure your DeepSeek-R1 model settings"
echo "2. Create your API service"
echo "3. DeepSeek-R1 code is available in ~/deepseek-app/DeepSeek-R1"