# DeepSeek-R1 HIE Integration

This repository contains code for running the DeepSeek-R1-Distill-Qwen-7B reasoning component as a microservice to filter and preprocess medical records before sending them to LLM APIs. It's designed to work with MAIA (Medical AI Assistant) and the NOSH patient record system.

## Repository Contents

- **app.py** - The FastAPI application that handles API requests and processes them through the DeepSeek model
- **setup-deepseek.sh** - Script to set up the environment, install dependencies, and prepare for model deployment
- **install.sh** - One-step installation script that runs setup and configures the system service
- **deepseek-optimize.sh** - Script to optimize DeepSeek for maximum performance
- **package.json** - NPM scripts for easy management and operation
- **README.md** - This documentation file

## Minimum Requirements

### Server Deployment
- Ubuntu 22.04 LTS or 24.04 LTS (Linux)
- macOS 12+ (Monterey or newer, with M-series chips for best performance)
- Minimum 8GB RAM (16GB recommended)
- At least 50GB storage

### Local Development
- 8-core CPU (Intel i7-9700K, Apple M1/M2/M3/M4, or equivalent)
- 32GB RAM recommended
- 20GB free disk space
- Windows, macOS, or Linux

**Note**: Even with recommended specifications, processing can take 10-30 seconds per request due to the size of the model. This is normal behavior for running a 7B parameter model on CPU. Apple Silicon M-series processors offer significantly better performance.

## Installation

The installation process has been streamlined with cross-platform support for both Linux and macOS systems.

### One-Step Installation

The easiest way to install is using the installation script:

```bash
# Install git (if not already installed)
# Ubuntu/Debian
sudo apt update && sudo apt install git -y
# macOS (with Homebrew)
brew install git

# Clone the repository
git clone https://github.com/abeuscher/deepseek-r1-hie.git
cd deepseek-r1-hie

# Run the installer
chmod +x install.sh
./install.sh
```

During the installation, you'll be prompted to set up SSL with Let's Encrypt if you have a domain name pointed to your server.

### NPM Scripts (Optional)

If you prefer using NPM scripts for easier management (requires Node.js):

```bash
# Install script
npm run install-full

# Start/Stop service
npm start
npm stop

# Check service status
npm run status
```

## Performance Optimization

After installation, you can optimize DeepSeek for your specific hardware:

```bash
# Run optimization script
chmod +x deepseek-optimize.sh
./deepseek-optimize.sh

# Or with NPM
npm run optimize
```

This will enhance performance with:
- Hardware-specific optimizations (Apple Silicon, CUDA, or CPU)
- Result caching for faster repeated queries
- Memory and thread optimizations
- API performance enhancements

## Using the API

Once running, the service provides a REST API with the following endpoints:

- `GET /health` - For checking service health
- `POST /process-context` - For preprocessing patient records
- `GET /cache/clear` - Clear the result cache

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
  // Show loading indicator - processing can take 10-30 seconds depending on hardware
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

## Troubleshooting

### Service Management

The most common commands for managing the service:

**Linux:**
```bash
# Check service status
sudo systemctl status deepseek

# View logs
sudo journalctl -u deepseek -f
```

**macOS:**
```bash
# Check service status
launchctl list | grep com.deepseek.api

# View logs
cat ~/deepseek-app/logs/deepseek.log
```

**Using NPM scripts (both platforms):**
```bash
# Check status
npm run status

# View logs
npm run logs
npm run logs:live
```

### Common Issues

- **504 Gateway Timeout**: If using Nginx, increase timeouts in your site configuration:
  ```
  proxy_connect_timeout 600;
  proxy_send_timeout 600;
  proxy_read_timeout 600;
  ```

- **CORS Errors**: The API includes CORS middleware, but ensure your Nginx configuration properly handles CORS headers

- **Slow Response Times**: Use the optimization script for better performance. Additionally:
  - Apple Silicon Macs offer significantly better performance than equivalent Intel systems
  - Use caching (built-in with the optimization script) for common queries
  - Consider hardware with more RAM and faster CPU/GPU

- **"command not found: python"**: On macOS, use `python3` instead of `python` or use the NPM scripts which handle this correctly

### Uninstalling

To completely remove the installation:

```bash
# Using NPM script
npm run uninstall

# Or manually
# On macOS
launchctl unload ~/Library/LaunchAgents/com.deepseek.api.plist
rm ~/Library/LaunchAgents/com.deepseek.api.plist
rm -rf ~/deepseek-app

# On Linux
sudo systemctl stop deepseek
sudo systemctl disable deepseek
sudo rm /etc/systemd/system/deepseek.service
sudo systemctl daemon-reload
rm -rf ~/deepseek-app
```

## License

DeepSeek-R1-Distill-Qwen-7B is licensed under the MIT License and is based on Qwen2.5-Math-7B, which is licensed under Apache 2.0 License.