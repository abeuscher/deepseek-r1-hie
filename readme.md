# DeepSeek-R1 HIE Integration

This repository contains code for running the DeepSeek-R1-Distill-Qwen-7B reasoning component as a microservice to filter and preprocess medical records before sending them to LLM APIs. It's designed to work with MAIA (Medical AI Assistant) and the NOSH patient record system.

## Quick Installation

Follow these steps to set up the DeepSeek-R1 reasoning service on a fresh Ubuntu server:

### Prerequisites

- Ubuntu 22.04 LTS or 24.04 LTS
- Minimum 8GB RAM (16GB recommended)
- At least 50GB storage

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

## JavaScript Integration

Here's how to integrate with JavaScript in your front-end application:

```javascript
// Using fetch API
async function processPatientContext(patientData, query) {
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
  
  return await response.json();
}

// Example usage
processPatientContext(patientRecord, "What medications is this patient taking?")
  .then(data => {
    console.log("Relevant context:", data.relevant_context);
    // Use this context in your LLM API call
  })
  .catch(error => {
    console.error("Error:", error);
  });
```

## System Architecture

This service acts as a preprocessing layer that:
1. Receives patient record data and a specific query
2. Uses DeepSeek-R1-Distill-Qwen-7B's reasoning capabilities to identify relevant portions of the record
3. Returns only the contextually important information to be included in prompts to LLM APIs

This approach allows handling large patient records without exceeding token limits in main LLM services like Anthropic Claude or ChatGPT.

## Troubleshooting

If you encounter any issues:

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

If the service fails to start due to memory constraints, check that the swap file was created:
```bash
swapon --show
```

## License

DeepSeek-R1-Distill-Qwen-7B is licensed under the MIT License and is based on Qwen2.5-Math-7B, which is licensed under Apache 2.0 License.
