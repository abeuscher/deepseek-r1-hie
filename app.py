from fastapi import FastAPI, HTTPException, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import uvicorn
import os
import logging
import asyncio
import time
import functools
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
MODEL_PATH = os.path.join(os.path.expanduser("~"), "deepseek-app", "models", "DeepSeek-R1-Distill-Qwen-7B")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95

# Performance optimization settings
CACHE_DIR = os.path.join(os.path.expanduser("~"), "deepseek-app", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Detect available devices
mps_available = torch.backends.mps.is_available()
cuda_available = torch.cuda.is_available()

# Sharding configuration
# These settings control how the model is split between GPU and CPU
GPU_LAYERS = 4      # Number of transformer layers to keep on GPU (adjust based on memory)
SHARD_SIZE = 1      # Shard embedding weights if set to 1

if mps_available:
    MAIN_DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logger.info("Using Apple Silicon GPU (MPS) with model sharding")
elif cuda_available:
    MAIN_DEVICE = "cuda"
    logger.info("Using NVIDIA GPU (CUDA) with model sharding")
else:
    MAIN_DEVICE = "cpu"
    logger.info("No GPU detected, using CPU only")
    
# Optimize CPU threads for CPU portions
torch.set_num_threads(max(1, os.cpu_count() - 1))
logger.info(f"Set PyTorch to use {torch.get_num_threads()} CPU threads")

# Helper function for caching
def get_cache_key(patient_data, query, max_length):
    """Generate a unique cache key based on inputs"""
    key_data = {
        "patient_data": patient_data,
        "query": query,
        "max_length": max_length,
        "model": MODEL_NAME
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

# Clean up memory
def clean_memory():
    """Force garbage collection and clear CUDA cache if available"""
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()

# Run CPU-intensive tasks in a thread pool
def run_in_threadpool(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )
    return wrapper

# Create device map for model sharding
def create_device_map():
    """Create a device map to distribute model layers between GPU and CPU"""
    device_map = {}
    
    # Critical components on GPU
    device_map["model.embed_tokens"] = MAIN_DEVICE
    device_map["model.norm"] = MAIN_DEVICE
    device_map["lm_head"] = MAIN_DEVICE
    
    # First few transformer layers on GPU for better performance
    # Adjust GPU_LAYERS based on available memory
    total_layers = 32  # Typical for a 7B model, will be adjusted if needed
    
    for i in range(total_layers):
        try:
            if i < GPU_LAYERS:
                device_map[f"model.layers.{i}"] = MAIN_DEVICE
            else:
                device_map[f"model.layers.{i}"] = "cpu"
        except:
            # This would happen if we've exceeded the actual number of layers
            # We can break and proceed with what we have
            break
    
    logger.info(f"Created device map with {GPU_LAYERS} layers on {MAIN_DEVICE}")
    return device_map

# Models are loaded at startup and shutdown at teardown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer on startup
    logger.info(f"Loading model {MODEL_NAME}...")
    
    # Clean up memory before loading
    clean_memory()
    
    logger.info(f"Downloading model from Hugging Face: {MODEL_NAME}")
    # Ensure the directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Download tokenizer
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully")
    
    # Create sharded device map
    device_map = create_device_map()
    
    # Load model with sharding
    logger.info("Loading model with CPU/GPU sharding...")
    try:
        app.state.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device_map,          # Custom device map for sharding
            torch_dtype=torch.float16,      # Half precision to reduce memory
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded successfully with sharding!")
        logger.info(f"Model is using devices: {app.state.model.hf_device_map}")
        
    except Exception as e:
        logger.error(f"Error loading model with sharding: {str(e)}")
        logger.warning("Attempting to load model with reduced settings...")
        
        # Fallback to even simpler loading with fewer GPU layers
        global GPU_LAYERS
        GPU_LAYERS = 2  # Reduce GPU layers for retry
        device_map = create_device_map()
        
        try:
            app.state.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("Model loaded with reduced GPU usage!")
        except Exception as e2:
            logger.error(f"Failed to load model: {str(e2)}")
            raise
    
    # Clean up after loading
    clean_memory()
    
    yield
    
    # Clean up on shutdown
    logger.info("Shutting down and cleaning up...")
    del app.state.model
    del app.state.tokenizer
    clean_memory()

# Add performance middleware
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 2.0:  # Adjust threshold as needed
        logger.warning(f"Slow request to {request.url.path}: {process_time:.2f}s")
        
    return response

app = FastAPI(title="DeepSeek-R1 Reasoning API", lifespan=lifespan)
app.middleware("http")(add_process_time_header)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add response compression
try:
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    logger.info("GZip compression enabled")
except ImportError:
    logger.info("GZip middleware not available")

class ContextRequest(BaseModel):
    """Request model for context processing."""
    patient_data: Dict[str, Any] = Field(..., description="The full patient record data")
    query: str = Field(..., description="The user's query about the patient")
    max_context_length: Optional[int] = Field(1000, description="Maximum number of tokens for the returned context")

class ContextResponse(BaseModel):
    """Response model with filtered context."""
    relevant_context: str = Field(..., description="The extracted relevant context from the patient data")
    reasoning: Optional[str] = Field(None, description="Optional explanation of the reasoning process")
    cached: Optional[bool] = Field(None, description="Indicates if the result was retrieved from cache")

@run_in_threadpool
def extract_context(model, tokenizer, patient_data, query, max_length=1000):
    """
    Use the DeepSeek model to extract relevant context from patient data.
    """
    # Format the input for the model
    prompt = f"""<think>
I need to analyze a patient's medical record to extract only the information relevant to the following question:
"{query}"

Here is the complete patient record:
{json.dumps(patient_data, indent=2)}

I will extract only the sections, symptoms, diagnoses, treatments, lab results, and other information that are directly relevant to answering the specific question. I will ignore irrelevant information.
</think>

Based on the medical record, here is the relevant information to answer the question:
"""
    
    # Generate response from the model
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Make sure input is on the same device as the first module 
    # This is especially important for sharded models
    first_param = next(model.parameters())
    first_device = first_param.device
    input_ids = input_ids.to(first_device)
    
    # Create an attention mask to avoid warnings
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        # Clean memory before generation
        clean_memory()
        
        try:
            # Limit generation length to prevent memory issues
            actual_max_tokens = min(max_length, 800)
            
            # For sharded models, avoid using autocast which can cause issues
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=actual_max_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True
            )
            
            # Decode the output and extract the relevant context
            full_output = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            relevant_context = full_output[len(prompt):].strip()
            
            return {
                "context": relevant_context,
                "reasoning": None  # Could extract reasoning from <think> tags if present in output
            }
        finally:
            # Clean memory after generation
            clean_memory()

@app.post("/process-context", response_model=ContextResponse)
async def process_context(request: ContextRequest, background_tasks: BackgroundTasks):
    """
    Process patient data and extract relevant context for a specific query.
    """
    try:
        # Check cache first
        cached_result = get_cached_result(
            request.patient_data, 
            request.query, 
            request.max_context_length
        )
        
        if cached_result:
            # Add cached flag for monitoring
            cached_result["cached"] = True
            return ContextResponse(
                relevant_context=cached_result["context"],
                reasoning=cached_result.get("reasoning"),
                cached=True
            )
        
        # Use the DeepSeek model to extract relevant context
        result = await extract_context(
            app.state.model,
            app.state.tokenizer,
            request.patient_data,
            request.query,
            max_length=request.max_context_length
        )
        
        # Cache the result in the background
        background_tasks.add_task(
            save_to_cache, 
            request.patient_data, 
            request.query, 
            request.max_context_length, 
            result
        )
        
        return ContextResponse(
            relevant_context=result["context"],
            reasoning=result.get("reasoning"),
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
    
    # Get memory info based on device
    memory_info = {
        "cpu_threads": torch.get_num_threads(),
        "processor_count": os.cpu_count()
    }
    
    # Get free RAM using system command
    try:
        import subprocess
        mem_info = subprocess.check_output(['vm_stat']).decode('utf-8')
        for line in mem_info.split('\n'):
            if 'Pages free' in line:
                free_pages = int(line.split(':')[1].strip()[:-1]) * 4096
                memory_info["free_memory_mb"] = free_pages / (1024 * 1024)
    except:
        pass
    
    # Add model sharding info if available
    sharding_info = {}
    if hasattr(app.state, "model") and hasattr(app.state.model, "hf_device_map"):
        sharding_info = dict(app.state.model.hf_device_map)
    
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "main_device": MAIN_DEVICE,
        "gpu_layers": GPU_LAYERS,
        "memory_info": memory_info,
        "cache_entries": len(cache_files),
        "sharding_info": sharding_info
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

@app.get("/adjust-sharding/{gpu_layers}")
async def adjust_sharding(gpu_layers: int):
    """Dynamically adjust model sharding (for experimentation)"""
    # NOTE: This requires restarting the application to take effect
    try:
        global GPU_LAYERS
        original_layers = GPU_LAYERS
        GPU_LAYERS = max(1, min(24, gpu_layers))  # Clamp between 1 and 24
        
        return {
            "status": "success", 
            "message": f"GPU layers adjusted from {original_layers} to {GPU_LAYERS}. Restart the application for changes to take effect."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adjusting sharding: {str(e)}")

if __name__ == "__main__":
    # Use localhost if behind a reverse proxy like Nginx
    host = "127.0.0.1" if os.path.exists("/etc/nginx/sites-enabled/deepseek") else "0.0.0.0"
    uvicorn.run("app:app", host=host, port=8000, log_level="info", reload=False)