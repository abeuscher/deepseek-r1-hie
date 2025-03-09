from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import uvicorn
import os
import sys
import logging
import time
import signal
import subprocess
from contextlib import asynccontextmanager
import hashlib
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("deepseek-api")

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.6
TOP_P = 0.95

# Simple cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), "deepseek-app", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# CRITICAL: Completely disable GPU and MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_ENABLE_FALLBACK"] = "0"

# Explicitly disable MPS availability detection
def disable_mps():
    if hasattr(torch.backends, "mps"):
        # Use a direct monkeypatch approach to ensure MPS is never used
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
        # Set these for older PyTorch versions
        if hasattr(torch, "has_mps"):
            torch.has_mps = False
    logger.info("MPS (Metal Performance Shaders) forcibly disabled")

# Call the function to disable MPS
disable_mps()

# Set device to CPU explicitly
DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

# Set reasonable thread count for CPU
NUM_THREADS = max(1, min(os.cpu_count() - 1, 4))  # Cap at 4 threads to avoid overuse
torch.set_num_threads(NUM_THREADS)
logger.info(f"Set PyTorch to use {NUM_THREADS} CPU threads")

# Function to terminate any process using port 8000
def kill_process_on_port(port=8000):
    """Find and kill any process using the specified port"""
    try:
        # For macOS
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            pids = result.stdout.strip().split("\n")
            logger.info(f"Found processes using port {port}: {pids}")
            
            for pid in pids:
                try:
                    pid = int(pid.strip())
                    # Don't kill our own process
                    if pid != os.getpid():
                        logger.info(f"Killing process {pid} using port {port}")
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.5)  # Give it a moment to terminate
                        # Check if it's still running and force kill if needed
                        try:
                            os.kill(pid, 0)  # This will raise an error if process is gone
                            logger.info(f"Process {pid} still alive, sending SIGKILL")
                            os.kill(pid, signal.SIGKILL)
                        except OSError:
                            logger.info(f"Process {pid} terminated successfully")
                except (ValueError, ProcessLookupError) as e:
                    logger.warning(f"Error processing PID {pid}: {str(e)}")
            
            # Verify port is now free
            time.sleep(1)  # Wait a bit for the OS to release the port
            check = subprocess.run(
                ["lsof", "-i", f":{port}", "-t"], 
                capture_output=True, 
                text=True
            )
            if check.stdout.strip():
                logger.warning(f"Port {port} still in use after termination attempts")
            else:
                logger.info(f"Port {port} is now free")
        else:
            logger.info(f"No process found using port {port}")
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {str(e)}")

# Simple caching functions
def get_cache_key(patient_data, query, max_length):
    """Generate a unique cache key based on inputs"""
    key_data = {
        "patient_data": patient_data,
        "query": query,
        "max_length": max_length,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached_result(patient_data, query, max_length):
    """Try to retrieve a cached result"""
    cache_key = get_cache_key(patient_data, query, max_length)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                logger.info(f"Cache hit for query: {query[:50]}...")
                return result
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
    
    return None

def save_to_cache(patient_data, query, max_length, result):
    """Save result to cache"""
    cache_key = get_cache_key(patient_data, query, max_length)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        logger.warning(f"Failed to save to cache: {str(e)}")

# Memory management
def clean_memory():
    """Force garbage collection"""
    gc.collect()
    logger.info("Memory cleaned")

# App lifespan for model loading/unloading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up on startup
    clean_memory()
    
    # Load model and tokenizer on startup
    logger.info(f"Loading model {MODEL_NAME}...")
    
    # Verify we're using CPU
    if torch.cuda.is_available():
        logger.warning("CUDA is available but we're forcing CPU mode for stability")
    
    # Double-check MPS is disabled
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.warning("MPS is still reporting as available, attempting to disable again")
        disable_mps()
    
    # Download tokenizer
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully")
    
    # Load model with minimal parameters and in simple CPU mode
    logger.info("Loading model in CPU mode...")
    try:
        # Force CPU and set dtype
        torch_dtype = torch.float32
        logger.info(f"Using dtype: {torch_dtype}")
        
        app.state.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"  # Explicitly set to CPU
        )
        
        # Ensure model is on CPU by iterating through all parameters
        for param in app.state.model.parameters():
            if param.device.type != "cpu":
                logger.warning(f"Found parameter on {param.device}, moving to CPU")
                param.data = param.data.to("cpu")
        
        # Check embedding layer specifically
        if hasattr(app.state.model.model, "embed_tokens"):
            if app.state.model.model.embed_tokens.weight.device.type != "cpu":
                logger.warning(f"Embedding layer on {app.state.model.model.embed_tokens.weight.device}, moving to CPU")
                app.state.model.model.embed_tokens = app.state.model.model.embed_tokens.to("cpu")
        
        logger.info("Model loaded successfully on CPU!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    clean_memory()
    yield
    
    # Clean up on shutdown
    logger.info("Shutting down and cleaning up...")
    del app.state.model
    del app.state.tokenizer
    clean_memory()

# Simple performance middleware
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    if process_time > 2.0:
        logger.warning(f"Slow request to {request.url.path}: {process_time:.2f}s")
        
    return response

app = FastAPI(title="DeepSeek-R1 Reasoning API", lifespan=lifespan)
app.middleware("http")(add_process_time_header)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContextRequest(BaseModel):
    """Request model for context processing."""
    patient_data: Dict[str, Any] = Field(..., description="The patient record data")
    query: str = Field(..., description="The user's query about the patient")
    max_context_length: Optional[int] = Field(500, description="Maximum number of tokens for the returned context")

class ContextResponse(BaseModel):
    """Response model with filtered context."""
    relevant_context: str = Field(..., description="The extracted relevant context from the patient data")
    cached: Optional[bool] = Field(None, description="Indicates if the result was retrieved from cache")

# Simplified patient data processing
def process_large_patient_data(patient_data):
    """Break down large patient data to fit in context window"""
    data_str = json.dumps(patient_data)
    
    # If data is small enough, return as is
    if len(data_str) < 50000:  # Reduced to 50KB for safety
        return patient_data
    
    # Otherwise truncate
    logger.warning("Patient data is very large, truncating to fit in context window")
    truncated_str = data_str[:50000] + "... [truncated due to size]"
    
    try:
        return json.loads(truncated_str)
    except:
        return {"data": "Patient data was too large and was truncated"}

def extract_context(model, tokenizer, patient_data, query, max_length=500):
    """Use the DeepSeek model to extract relevant context from patient data."""
    # Process large data
    processed_patient_data = process_large_patient_data(patient_data)
    
    # Format the input
    prompt = f"""<think>
I need to analyze a patient's medical record to extract only the information relevant to the following question:
"{query}"

Here is the complete patient record:
{json.dumps(processed_patient_data, indent=2)}

I will extract only the sections, symptoms, diagnoses, treatments, lab results, and other information that are directly relevant to answering the specific question. I will ignore irrelevant information.
</think>

Based on the medical record, here is the relevant information to answer the question:
"""
    
    # Check we're using CPU
    device_to_use = "cpu"
    
    # Generate response 
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device_to_use)
    attention_mask = torch.ones_like(input_ids)
    
    logger.info(f"Input device: {input_ids.device}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    with torch.no_grad():
        clean_memory()
        
        try:
            # Limit generation length for safety
            actual_max_tokens = min(max_length, 500)
            
            # Verify model's embedding layer is on CPU
            if hasattr(model.model, "embed_tokens"):
                embed_device = model.model.embed_tokens.weight.device
                logger.info(f"Embedding layer device: {embed_device}")
                
                if embed_device.type != "cpu":
                    logger.warning(f"Moving embedding layer from {embed_device} to CPU")
                    model.model.embed_tokens = model.model.embed_tokens.to("cpu")
            
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=actual_max_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the output
            full_output = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            relevant_context = full_output[len(prompt):].strip()
            
            return {"context": relevant_context}
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            # Try a simplified response if generation fails
            return {"context": f"Error processing patient data: {str(e)}. Please try with a simpler query or less data."}
        finally:
            clean_memory()

@app.post("/process-context", response_model=ContextResponse)
async def process_context(request: ContextRequest):
    """Process patient data and extract relevant context for a specific query."""
    try:
        # Check cache first
        cached_result = get_cached_result(
            request.patient_data, 
            request.query, 
            request.max_context_length
        )
        
        if cached_result:
            return ContextResponse(
                relevant_context=cached_result["context"],
                cached=True
            )
        
        # Use the DeepSeek model
        result = extract_context(
            app.state.model,
            app.state.tokenizer,
            request.patient_data,
            request.query,
            max_length=request.max_context_length
        )
        
        # Save to cache directly (no background task)
        save_to_cache(
            request.patient_data, 
            request.query, 
            request.max_context_length, 
            result
        )
        
        return ContextResponse(
            relevant_context=result["context"],
            cached=False
        )
    except Exception as e:
        logger.error(f"Error processing context: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing context: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    # Get cache stats
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    # Get memory info
    memory_info = {
        "cpu_threads": torch.get_num_threads(),
        "processor_count": os.cpu_count()
    }
    
    # Check model device if loaded
    model_device = "not_loaded"
    if hasattr(app.state, "model"):
        try:
            model_device = next(app.state.model.parameters()).device
        except:
            model_device = "unknown"
    
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": str(model_device),
        "memory_info": memory_info,
        "cache_entries": len(cache_files),
    }

@app.get("/cache/clear")
async def clear_cache():
    """Clear the result cache"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
        count = len(cache_files)
        
        for file in cache_files:
            os.remove(os.path.join(CACHE_DIR, file))
            
        return {"status": "success", "cleared_entries": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

if __name__ == "__main__":
    # Kill any existing process using port 8000
    logger.info("Checking for existing processes using port 8000")
    kill_process_on_port(8000)
    
    # Use localhost binding
    host = "0.0.0.0"  # Bind to all interfaces
    port = 8000
    
    try:
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run("app:app", host=host, port=port, log_level="info", reload=False)
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.error(f"Port {port} is still in use despite termination attempts")
            sys.exit(1)
        else:
            raise