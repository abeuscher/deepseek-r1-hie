# DeepSeek-R1 HIE Integration

This repository contains code for running the DeepSeek-R1-Distill-Qwen-7B reasoning component as a microservice to filter and preprocess medical records before sending them to LLM APIs. It's designed to work with MAIA (Medical AI Assistant) and the NOSH patient record system.

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
   This will install all required dependencies and prepare the environment.

4. Copy the API service files:
   ```bash
   cp app.py ~/deepseek-app/api/
   ```

5. Set up as a system service (optional but recommended):
   ```bash
   sudo cp deepseek.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable deepseek
   sudo systemctl start deepseek
   ```

6. Alternatively, start the API manually:
   ```bash
   cd ~/deepseek-app
   source venv/bin/activate
   cd api
   python app.py
   ```

### Using the API

Once running, the service provides a REST API endpoint at:
- `http://your-server-ip:8000/process-context` - For preprocessing patient records
- `http://your-server-ip:8000/health` - For checking service health

Example curl request:
```bash
curl -X POST "http://your-server-ip:8000/process-context" \
     -H "Content-Type: application/json" \
     -d '{
           "patient_data": {"medical_history": {"...": "..."}},
           "query": "What medications is this patient taking for hypertension?",
           "max_context_length": 1000
         }'
```

## System Architecture

This service acts as a preprocessing layer that:
1. Receives patient record data and a specific query
2. Uses DeepSeek-R1-Distill-Qwen-7B's reasoning capabilities to identify relevant portions of the record
3. Returns only the contextually important information to be included in prompts to LLM APIs

This approach allows handling large patient records without exceeding token limits in the main LLM services like Anthropic Claude or ChatGPT.

## Model Information

This service uses the DeepSeek-R1-Distill-Qwen-7B model, which is:
- A distilled version of DeepSeek-R1's reasoning capabilities 
- Optimized to run on standard hardware (7B parameters)
- Specifically designed for complex reasoning tasks
- Capable of running on a server with 16GB RAM

## Troubleshooting

If you encounter any issues during setup:
- Check the system logs: `sudo journalctl -u deepseek -f`
- Ensure your server meets the minimum hardware requirements
- Verify network connectivity for API access and model downloading

## License

DeepSeek-R1-Distill-Qwen-7B is licensed under the MIT License and is based on Qwen2.5-Math-7B, which is licensed under Apache 2.0 License.
