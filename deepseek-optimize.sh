#!/bin/bash
# DeepSeek R1 Optimization Script
# This script optimizes an existing DeepSeek R1 installation for maximum performance
# Run this after completing the standard installation
#
# Usage: bash deepseek-optimize.sh

set -e  # Exit on error

echo "=== Starting DeepSeek R1 Performance Optimization ==="

# Configuration - modify these as needed
APP_DIR=~/deepseek-app
MODEL_CACHE_DIR=~/deepseek-app/models
ENABLE_GPU=true
QUANTIZE_MODEL=true

# Detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
  OS_TYPE="macos"
  echo "Detected macOS operating system"
  
  # Check for Apple Silicon
  if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon (M-series) processor"
    IS_M_SERIES=true
  else
    echo "Detected Intel architecture"
    IS_M_SERIES=false
  fi
else
  OS_TYPE="linux"
  echo "Detected Linux operating system"
  IS_M_SERIES=false
fi

# Create optimization directory
mkdir -p $APP_DIR/optimization

# Function to create a Python file with optimizations
create_optimization_module() {
  echo "Creating optimization module..."
  cat > $APP_DIR/optimization/optimize.py << EOF
# DeepSeek R1 Optimization Module
import os
import torch
import hashlib
import pickle
from pathlib import Path

# Environment optimizations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable Metal Performance Shaders fallback
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() - 1))  # Optimal thread count

# Create cache directory
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)

# Document processing cache
def get_document_hash(document_text):
    """Generate a hash for a document to use as cache key"""
    return hashlib.md5(document_text.encode()).hexdigest()

def get_query_hash(query):
    """Generate a hash for a query to use as cache key"""
    return hashlib.md5(query.encode()).hexdigest() 

def cache_result(document_text, query, result):
    """Cache the result of processing a document with a query"""
    doc_hash = get_document_hash(document_text)
    query_hash = get_query_hash(query)
    cache_path = cache_dir / f"{doc_hash}_{query_hash}.pkl"
    
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    
    return str(cache_path)

def get_cached_result(document_text, query):
    """Retrieve cached result if available"""
    doc_hash = get_document_hash(document_text)
    query_hash = get_query_hash(query)
    cache_path = cache_dir / f"{doc_hash}_{query_hash}.pkl"
    
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

# Model loading optimizations
def get_optimized_model_config(model_name, quantize=True):
    """Get optimized configuration for loading the model"""
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name)
    
    # Performance settings
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "flash_attention_2"
    
    return config

def load_optimized_model(model_name, quantize=True, device=None):
    """Load model with optimized settings"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    # Determine the device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Prepare quantization config if needed
    if quantize and (device == "cuda" or device == "cpu"):
        # 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = None
    
    # Load the model with optimizations
    model_config = get_optimized_model_config(model_name, quantize)
    
    # Model loading parameters
    kwargs = {
        "pretrained_model_name_or_path": model_name,
        "config": model_config,
        "device_map": "auto" if device != "mps" else device,
        "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
    }
    
    # Add quantization if available
    if quantization_config:
        kwargs["quantization_config"] = quantization_config
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    
    return model, device

# Batch processing for multiple documents
def process_documents_batch(model, tokenizer, documents, query, max_batch_size=4):
    """Process multiple documents in batches for better performance"""
    results = []
    
    for i in range(0, len(documents), max_batch_size):
        batch = documents[i:i+max_batch_size]
        batch_results = []
        
        for doc in batch:
            # Try cache first
            cached = get_cached_result(doc, query)
            if cached:
                batch_results.append(cached)
                continue
                
            # Process and cache
            result = process_single_document(model, tokenizer, doc, query)
            cache_result(doc, query, result)
            batch_results.append(result)
            
        results.extend(batch_results)
    
    return results

# Context window optimization
def optimize_context_length(text, max_length, query):
    """Optimize text to fit within context window prioritizing relevance to query"""
    # Simple optimization: prioritize chunks containing query terms
    import re
    from nltk.tokenize import sent_tokenize
    
    # Ensure NLTK data is available
    try:
        sent_tokenize("Test sentence.")
    except LookupError:
        import nltk
        nltk.download('punkt', quiet=True)
    
    # Break into sentences
    sentences = sent_tokenize(text)
    
    # Score sentences by relevance to query
    query_terms = set(re.findall(r'\\w+', query.lower()))
    
    scored_sentences = []
    for sentence in sentences:
        sentence_terms = set(re.findall(r'\\w+', sentence.lower()))
        overlap = len(query_terms.intersection(sentence_terms))
        scored_sentences.append((sentence, overlap))
    
    # Sort by relevance score (descending)
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Reconstruct text prioritizing relevant sentences
    optimized_text = " ".join([s[0] for s in scored_sentences])
    
    # Truncate if still too long
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Fast tokenizer for length estimation
    if len(tokenizer.encode(optimized_text)) > max_length:
        encoded = tokenizer.encode(optimized_text, truncation=True, max_length=max_length)
        optimized_text = tokenizer.decode(encoded)
    
    return optimized_text
EOF

  echo "Created optimization module at $APP_DIR/optimization/optimize.py"
}

# Create Transformer model helpers
create_model_helpers() {
  echo "Creating model helper utilities..."
  cat > $APP_DIR/optimization/model_utils.py << EOF
# DeepSeek R1 Model Utilities
import torch
from pathlib import Path
import os

def get_optimal_device():
    """Determine the optimal device for model inference"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"

def optimize_for_device(device):
    """Set environment variables and configurations for optimal performance on the device"""
    if device == "mps":
        # For Apple Silicon
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif device == "cuda":
        # For NVIDIA GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    
    # Common optimizations
    torch.set_float32_matmul_precision('high')
EOF

  echo "Created model utilities at $APP_DIR/optimization/model_utils.py"
}

# Create API performance optimization helper
create_api_optimization() {
  echo "Creating API optimization utilities..."
  cat > $APP_DIR/optimization/api_optimize.py << EOF
# DeepSeek R1 API Optimization
import asyncio
import functools
import time
from fastapi import FastAPI, BackgroundTasks

# Performance monitoring middleware
async def performance_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests (adjust threshold as needed)
    if process_time > 1.0:
        endpoint = request.url.path
        print(f"SLOW REQUEST: {endpoint} took {process_time:.2f} seconds")
    
    return response

# Apply optimizations to a FastAPI app
def optimize_api(app: FastAPI):
    # Add performance monitoring
    app.middleware("http")(performance_middleware)
    
    # Enable response compression
    try:
        from fastapi.middleware.gzip import GZipMiddleware
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    except ImportError:
        print("GZip middleware not available. Install with: pip install fastapi[standard]")
    
    # Return the optimized app
    return app

# Run CPU-intensive tasks in a thread pool to avoid blocking
def run_in_threadpool(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )
    return wrapper
EOF

  echo "Created API optimization utilities at $APP_DIR/optimization/api_optimize.py"
}

# Update the app to use optimizations
create_app_optimization_patch() {
  echo "Creating the optimization patch script..."
  cat > $APP_DIR/optimization/patch_app.py << EOF
import os
import sys
import re

def patch_app(app_path):
    """Patch the app.py file to include optimizations"""
    if not os.path.exists(app_path):
        print(f"Error: Could not find app file at {app_path}")
        return False
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Create backup
    with open(f"{app_path}.bak", 'w') as f:
        f.write(content)
    
    # Add imports
    import_patch = """
# Performance optimization imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization.optimize import load_optimized_model, cache_result, get_cached_result
from optimization.model_utils import get_optimal_device, optimize_for_device
from optimization.api_optimize import optimize_api, run_in_threadpool
"""
    
    if "from fastapi import" in content:
        content = re.sub(
            r'from fastapi import (.*)', 
            r'from fastapi import \1\n' + import_patch, 
            content
        )
    else:
        # Just add imports at the top after any other imports
        lines = content.split('\n')
        import_section_end = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_section_end = i + 1
        
        content = '\n'.join(lines[:import_section_end]) + import_patch + '\n'.join(lines[import_section_end:])
    
    # Patch FastAPI initialization
    if "app = FastAPI()" in content:
        content = content.replace(
            "app = FastAPI()", 
            "app = FastAPI()\n# Apply API optimizations\napp = optimize_api(app)"
        )
    
    # Patch model loading (adjust based on actual code)
    model_loading_patterns = [
        (r'model\s*=\s*AutoModelForCausalLM\.from_pretrained\((.*?)\)', 
         r'# Optimized model loading\ndevice = get_optimal_device()\noptimize_for_device(device)\nmodel, _ = load_optimized_model(\1)'),
    ]
    
    for pattern, replacement in model_loading_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Add caching to document processing endpoints
    if "/process-context" in content and "def process_context" in content:
        # Add caching to the processing function
        content = re.sub(
            r'async def process_context\((.*?)\):(.*?)return\s*(.*?)$',
            r'async def process_context(\1):\2# Check cache first\n    cached = get_cached_result(document, query)\n    if cached:\n        return cached\n\n    result = \3\n    # Cache the result\n    cache_result(document, query, result)\n    return result',
            content, 
            flags=re.DOTALL
        )
    
    # Write modified content
    with open(app_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {app_path} with optimizations")
    print(f"A backup was saved to {app_path}.bak")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        app_path = sys.argv[1]
    else:
        app_path = os.path.abspath("../api/app.py")
    
    success = patch_app(app_path)
    if success:
        print("App patched successfully!")
    else:
        print("Failed to patch app. See errors above.")
EOF

  echo "Created optimization patch script at $APP_DIR/optimization/patch_app.py"
}

# Create or update the initialization wrapper
create_init_module() {
  echo "Creating initialization module..."
  mkdir -p $APP_DIR/optimization
  
  cat > $APP_DIR/optimization/__init__.py << EOF
# DeepSeek R1 Optimization Package
from .optimize import load_optimized_model, cache_result, get_cached_result
from .model_utils import get_optimal_device, optimize_for_device
from .api_optimize import optimize_api, run_in_threadpool

__all__ = [
    'load_optimized_model', 'cache_result', 'get_cached_result',
    'get_optimal_device', 'optimize_for_device',
    'optimize_api', 'run_in_threadpool'
]
EOF

  echo "Created initialization module at $APP_DIR/optimization/__init__.py"
}

# Update the virtual environment with additional packages
update_venv() {
  echo "Installing additional optimization packages..."
  source $APP_DIR/venv/bin/activate
  
  # Install performance-related packages
  pip install nltk einops flash-attn --no-build-isolation || echo "Warning: Some packages could not be installed, but the optimization will still work."
  
  deactivate
}

# Create system optimization script for macOS
if [ "$OS_TYPE" = "macos" ]; then
  echo "Creating macOS system optimization script..."
  cat > $APP_DIR/optimize-macos-system.sh << EOF
#!/bin/bash
# DeepSeek R1 macOS System Optimization
# Run with: sudo bash optimize-macos-system.sh

if [ "\$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Please use: sudo bash \$0"
  exit 1
fi

echo "=== Optimizing macOS system for DeepSeek R1 ==="

# Disable Spotlight indexing
echo "Disabling Spotlight indexing..."
mdutil -i off /

# Increase disk cache for filesystem
echo "Optimizing filesystem cache..."
sysctl -w kern.maxvnodes=200000

# Disable sleep
echo "Disabling system sleep..."
pmset -a disablesleep 1
pmset -a sleep 0

# Set process priority for Python processes
cat > /Library/LaunchDaemons/ai.deepseek.priority.plist << EOL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.deepseek.priority</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>while true; do for pid in \$(pgrep -f "python.*app.py"); do renice -n -20 -p \$pid; done; sleep 60; done</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOL

chmod 644 /Library/LaunchDaemons/ai.deepseek.priority.plist
launchctl load -w /Library/LaunchDaemons/ai.deepseek.priority.plist

echo "System optimization complete!"
EOF

  chmod +x $APP_DIR/optimize-macos-system.sh
  echo "Created macOS system optimization script at $APP_DIR/optimize-macos-system.sh"
fi

# Apply optimization to the app
apply_optimizations() {
  echo "Applying optimizations to the application..."
  
  # Create cache directory
  mkdir -p $APP_DIR/cache
  
  # Check if app.py exists
  if [ -f "$APP_DIR/api/app.py" ]; then
    # Create a backup of the original app.py
    cp $APP_DIR/api/app.py $APP_DIR/api/app.py.original
    
    # Run the optimization patch
    cd $APP_DIR/optimization
    python3 patch_app.py $APP_DIR/api/app.py
  else
    echo "Warning: app.py not found at $APP_DIR/api/app.py"
    echo "After installing the application, run the optimization patch manually:"
    echo "  cd $APP_DIR/optimization"
    echo "  python3 patch_app.py /path/to/your/app.py"
  fi
}

# Main function
main() {
  # Create the optimization modules
  create_optimization_module
  create_model_helpers
  create_api_optimization
  create_init_module
  create_app_optimization_patch
  
  # Update virtual environment
  update_venv
  
  # Apply optimizations
  apply_optimizations
  
  echo "=== Optimization Complete ==="
  echo ""
  echo "DeepSeek R1 has been optimized for maximum performance."
  echo ""
  
  if [ "$OS_TYPE" = "macos" ]; then
    echo "For additional system-level optimizations, run:"
    echo "  sudo bash $APP_DIR/optimize-macos-system.sh"
    echo ""
  fi
  
  echo "To run the application with optimizations:"
  echo "  1. Start the service as normal:"
  echo "     launchctl load ~/Library/LaunchAgents/com.deepseek.api.plist"
  echo ""
  echo "  2. Or run manually with the virtual environment:"
  echo "     source $APP_DIR/venv/bin/activate"
  echo "     python3 $APP_DIR/api/app.py"
  echo ""
  echo "Performance monitoring is now enabled. Slow requests will be logged."
  echo "Document processing results are now cached for faster repeat queries."
}

# Run the main function
main