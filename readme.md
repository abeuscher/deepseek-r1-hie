# DeepSeek-R1 HIE Integration

This repository contains code for running the DeepSeek-R1 reasoning component as a microservice to filter and preprocess medical records before sending them to LLM APIs. It's designed to work with MAIA (Medical AI Assistant) and the NOSH patient record system.

## Quick Setup

Follow these steps to set up the DeepSeek-R1 reasoning service on a fresh Ubuntu server:

### Prerequisites

- Ubuntu 22.04 LTS or 24.04 LTS
- Minimum 8GB RAM (16GB recommended)
- At least 50GB storage

### Installation Steps

1. Install Git if not already installed:
   ```bash
   sudo apt update
   sudo apt install git -y
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/abeuscher/deepseek-r1-hie.git
   cd deepseek-r1-hie
   ```

3. Run the setup script:
   ```bash
   chmod +x setup-deepseek.sh
   ./setup-deepseek.sh
   ```
   This will install all required dependencies and the DeepSeek-R1 codebase.

4. Start the API service:
   ```bash
   cd ~/deepseek-app
   source venv/bin/activate
   python app.py
   ```

### Using the API

Once running, the service provides a REST API endpoint at:
- `http://your-server-ip:8000/process-context` - For preprocessing patient records
- `http://your-server-ip:8000/health` - For checking service health

See the API documentation for detailed usage instructions.

## System Architecture

This service acts as a preprocessing layer that:
1. Receives patient record data and a specific query
2. Uses DeepSeek-R1's reasoning capabilities to identify relevant portions of the record
3. Returns only the contextually important information to be included in prompts to LLM APIs

This approach allows handling large patient records without exceeding token limits in the main LLM services like Anthropic Claude or ChatGPT.

## Configuration

[Configuration documentation will be added after initial testing]

## Troubleshooting

If you encounter any issues during setup:
- Ensure your server meets the minimum hardware requirements
- Check that all dependencies installed correctly
- Verify network connectivity for API access

For specific error messages, please consult the troubleshooting guide or file an issue on this repository.

## License

[License information to be added]
