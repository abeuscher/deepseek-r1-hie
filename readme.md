# DeepSeek-R1 HIE Integration

This repository contains code for running the DeepSeek-R1-Distill-Qwen-7B reasoning component as a microservice to filter and preprocess medical records before sending them to LLM APIs. It's designed to work with MAIA (Medical AI Assistant) and the NOSH patient record system.

## Repository Contents

- **app.py** - The FastAPI application that handles API requests and processes them through the DeepSeek model
- **setup-deepseek.sh** - Script to set up the environment, install dependencies, and prepare for model deployment
- **install.sh** - One-step installation script that runs setup and configures the system service
- **deepseek.service** - Systemd service file for running the API as a background service on Linux
- **README.md** - This documentation file

## Minimum Requirements

### Server Deployment
- Ubuntu 22.04 LTS or 24.04 LTS
- Minimum 8GB RAM (16GB recommended)
- At least 50GB storage

### Local Development
- 8-core CPU (Intel i7-9700K or equivalent)
- 32GB RAM recommended
- 20GB free disk space
- Windows, macOS, or Linux

**Note**: Even with recommended specifications, processing can take 20-30 seconds per request due to the size of the model. This is normal behavior for running a 7B parameter model on CPU.

## Server Installation

Follow these steps to set up the DeepSeek-R1 reasoning service on a fresh Ubuntu server:

### One-Step Installation

The easiest way to install is using the installation script:

```bash
# Install git
sudo apt update
sudo apt install git -y

# Clone the repository
git clone https://github.com/abeuscher/deepseek-r1-hie.git
cd deepseek-r1-hie

# Run the installer
chmod +x install.sh
./install.sh
```

During the installation, you'll be prompted to set up SSL with Let's Encrypt if you have a domain name pointed to your server.

### Manual Installation

If you prefer to install step by step:

1. Run the environment setup script:
   ```bash
   chmod +x setup-deepseek.sh
   ./setup-deepseek.sh
   ```
   This will install all required dependencies and prepare the environment.

2. Copy the API service file:
   ```bash
   cp app.py ~/deepseek-app/api/
   ```

3. Start the service:
   ```bash
   sudo systemctl start deepseek
   ```

## Local Installation (Windows/macOS/Linux)

You can also run the service locally for development or testing:

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)

2. Clone the repository:
   ```
   git clone https://github.com/abeuscher/deepseek-r1-hie.git
   cd deepseek-r1-hie
   ```

3. Create and activate a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:
   ```
   pip install torch transformers fastapi uvicorn pydantic accelerate
   ```

5. Run the application:
   ```
   python app.py
   ```

The service will be available at http://localhost:8000. The first run will download the model, which may take some time.

## Using the API

Once running, the service provides a REST API with the following endpoints:

- `GET /health` - For checking service health
- `POST /process-context` - For preprocessing patient records

Example request to process a patient record:

```bash
curl -X POST "https://your-domain.com/process-context" \
     -H "Content-Type: application/json" \
     -d '{
           "patient_data": {"medical_history": {"...": "..."}},
           "query": "What medications is this patient taking for hypertension?",
           "max_context_length": 1000
         }'
```

For local testing, use:
```bash
curl -X POST "http://localhost:8000/process-context" \
     -H "Content-Type: application/json" \
     -d '{
           "patient_data": {"name":"Test Patient"},
           "query":"What is the patient name?",
           "max_context_length":100
         }'
```

## JavaScript Integration

Here's how to integrate with JavaScript in your front-end application:

```javascript
// Using fetch API
async function processPatientContext(patientData, query) {
  // Show loading indicator - important as processing can take 20-30 seconds
  showLoadingIndicator();
  
  try {
    const response = await fetch('https://your-domain.com/process-context', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        patient_data: patientData,
        query: query,
        max_context_length: 500
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    hideLoadingIndicator();
    return data;
  } catch (error) {
    hideLoadingIndicator();
    console.error("Error:", error);
    throw error;
  }
}

// Example usage
processPatientContext(patientRecord, "What medications is this patient taking?")
  .then(data => {
    console.log("Relevant context:", data.relevant_context);
    // Use this context in your LLM API call
  })
  .catch(error => {
    // Handle error
  });
```

## System Architecture

This service acts as a preprocessing layer that:
1. Receives patient record data and a specific query
2. Uses DeepSeek-R1-Distill-Qwen-7B's reasoning capabilities to identify relevant portions of the record
3. Returns only the contextually important information to be included in prompts to LLM APIs

This approach allows handling large patient records without exceeding token limits in main LLM services like Anthropic Claude or ChatGPT.

## Troubleshooting

### Server Deployment Issues

- Check the service status:
  ```bash
  sudo systemctl status deepseek
  ```

- View the logs:
  ```bash
  sudo journalctl -u deepseek -f
  ```

- Check available disk space:
  ```bash
  df -h
  ```

- Verify memory usage:
  ```bash
  free -h
  ```

- If the service fails to start due to memory constraints, check that the swap file was created:
  ```bash
  swapon --show
  ```

### Common Issues

- **504 Gateway Timeout**: If you're getting timeouts through Nginx, edit /etc/nginx/sites-available/deepseek to increase timeouts:
  ```
  proxy_connect_timeout 600;
  proxy_send_timeout 600;
  proxy_read_timeout 600;
  ```

- **CORS Errors**: Make sure your Nginx configuration correctly handles CORS headers, or access the API from the same domain

- **Slow Response Times**: This is normal for CPU inference. For production, consider:
  - Using a GPU-enabled machine
  - Pre-processing data in batches
  - Implementing caching for common queries

### Local Development Issues

- **Python Version Conflicts**: Ensure you're using Python 3.10+ and a clean virtual environment
- **Memory Errors**: Close other applications to free up RAM
- **Model Download Issues**: Ensure you have a stable internet connection and sufficient disk space

## License

DeepSeek-R1-Distill-Qwen-7B is licensed under the MIT License and is based on Qwen2.5-Math-7B, which is licensed under Apache 2.0 License.
