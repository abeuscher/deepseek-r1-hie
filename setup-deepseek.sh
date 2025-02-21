#!/bin/bash
# DeepSeek Environment Setup Script
# This script sets up the environment for the DeepSeek-R1-Distill-Qwen-7B reasoning component
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

# Upgrade pip and install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch==2.1.0 fastapi uvicorn requests 

# Install Hugging Face Transformers and related libraries
echo "Installing model dependencies..."
pip install transformers==4.38.0 accelerate sentencepiece protobuf

# Create the API service directory
mkdir -p $APP_DIR/api
cd $APP_DIR/api

# Create a model directory where we'll download the model
mkdir -p $APP_DIR/models

echo "=== Setup complete! ==="
echo "The environment has been prepared for DeepSeek-R1-Distill-Qwen-7B"
echo ""
echo "To activate the environment:"
echo "  cd ~/deepseek-app"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download the DeepSeek-R1-Distill-Qwen-7B model files"
echo "2. Start the API service"