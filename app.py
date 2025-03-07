from fastapi import FastAPI, HTTPException, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import torch
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

# Device strategy: "cpu", "gpu", or "hybrid"
DEVICE_STRATEGY = "hybrid"

# Performance optimization settings
CACHE_DIR = os.path.join(os.path.expanduser("~"), "deepseek-app", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure device based on strategy and available hardware
if torch.backends.mps.is_available() and DEVICE_STRATEGY != "cpu":
    # Enable Metal Performance Shaders optimizations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    if DEVICE_STRATEGY == "hybrid":
        logger.info("Using hybrid CPU+MPS strategy for optimal performance and memory usage")
        # We'll configure a hybrid device map later, but use MPS for input tensors
        DEVICE = "mps"
    else:
        logger.info("Using Apple Silicon GPU (MPS) for inference")
        DEVICE = "mps"
elif torch.cuda.is_available() and DEVICE_STRATEGY != "cpu":
    DEVICE = "cuda"
    # Enable TF32 for better performance on NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    
    if DEVICE_STRATEGY == "hybrid":
        logger.info("Using hybrid CPU+CUDA strategy for optimal performance and memory usage")
    else:
        logger.info("Using NVIDIA GPU (CUDA) for inference")
else:
    DEVICE = "cpu"
    # Optimize CPU threads
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    logger.info("Using CPU for inference")

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

# Background task to clean up GPU memory
async def cleanup_gpu_memory():
    """Clean up GPU memory after request processing"""
    if DEVICE == "cuda":
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

# Models are loaded at startup and shutdown at teardown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer on startup
    logger.info(f"Loading model {MODEL_NAME}...")
    
    logger.info(f"Downloading model from Hugging Face: {MODEL_NAME}")
    # Ensure the directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Download with memory-efficient settings
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Choose model loading strategy based on device strategy
    if DEVICE_STRATEGY == "hybrid" and (torch.backends.mps.is_available() or torch.cuda.is_available()):
        logger.info("Configuring hybrid device map for model layers...")
        
        # Configure device map for layer distribution
        device_map = {
            "model.embed_tokens": DEVICE,
            "model.norm": DEVICE,
            "lm_head": DEVICE
        }
        
        # Determine total number of layers (usually 24-32 for 7B models)
        total_layers = 32  # Conservative estimate, will be adjusted if needed
        
        # For MPS, keep a smaller number of layers on GPU to avoid memory issues
        # For CUDA, we can usually use more GPU layers
        if DEVICE == "mps":
            gpu_layers = 6  # Reduced from 8 to avoid memory issues on M1/M2/M4
        else:  # CUDA
            gpu_layers = 12
            
        # Distribute layers between GPU and CPU
        for i in range(total_layers):
            try:
                if i < gpu_layers:
                    # First layers on GPU
                    device_map[f"model.layers.{i}"] = DEVICE
                else:
                    # Remaining layers on CPU
                    device_map[f"model.layers.{i}"] = "cpu"
            except Exception as e:
                # This would happen if we've exceeded the actual number of layers
                # We can ignore this and move on
                break
                
        # Load model with the hybrid device map
        app.state.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
    else:
        # Standard loading based on device
        model_kwargs = {
            "pretrained_model_name_or_path": MODEL_NAME,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "offload_folder": "offload"
        }
        
        if DEVICE == "cuda":
            model_kwargs.update({
                "device_map": "auto",
                "torch_dtype": torch.float16
            })
        elif DEVICE == "mps":
            model_kwargs.update({
                "device_map": DEVICE,
                "torch_dtype": torch.float16
            })
        else:
            # CPU optimizations
            model_kwargs.update({
                "device_map": "auto"
            })
        
        app.state.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    logger.info("Model loaded successfully!")
    
    yield
    
    # Clean up on shutdown
    logger.info("Shutting down and cleaning up...")
    del app.state.model
    del app.state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    
    # Make sure input is on the same device as the model's first module
    # For hybrid setups, this should typically be GPU (embed_tokens)
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    
    try:
        with torch.no_grad():
            # For hybrid setups, don't use autocast as it can cause issues with split devices
            if DEVICE_STRATEGY == "hybrid":
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True
                )
            # For full GPU, use autocast
            elif DEVICE == "mps" or DEVICE == "cuda":
                with torch.autocast(device_type=DEVICE):
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        do_sample=True
                    )
            # For CPU, no autocast needed
            else:
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
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
    except RuntimeError as e:
        # If we encounter a memory error, fall back to CPU
        if "out of memory" in str(e):
            logger.warning("GPU out of memory. Falling back to CPU for this request.")
            # Move input to CPU
            input_ids = input_ids.to("cpu")
            
            # Temporarily move model to CPU for this generation
            device_map_backup = getattr(model, "hf_device_map", None)
            
            try:
                # Set all parameters to CPU temporarily
                model.to("cpu")
                
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        do_sample=True
                    )
                    
                full_output = tokenizer.decode(output[0], skip_special_tokens=True)
                relevant_context = full_output[len(prompt):].strip()
                
                return {
                    "context": relevant_context,
                    "reasoning": None
                }
            finally:
                # Restore device map if we had one
                if device_map_backup is not None and DEVICE_STRATEGY == "hybrid":
                    # This will restore the original device mapping
                    # In most cases a restart is cleaner, but this allows recovery
                    for name, device in device_map_backup.items():
                        try:
                            module = model.get_submodule(name)
                            module.to(device)
                        except:
                            pass
        
        # Re-raise other exceptions
        raise

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
        
        # Add cleanup task
        background_tasks.add_task(cleanup_gpu_memory)
        
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
    memory_info = {}
    
    # Get memory info based on device
    if DEVICE == "cuda" and torch.cuda.is_available():
        memory_info = {
            "total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3),
            "allocated": torch.cuda.memory_allocated(0) / (1024**3)
        }
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        # No direct memory query API for MPS, but we can add device_strategy
        memory_info = {
            "note": "MPS memory stats not available",
            "device_strategy": DEVICE_STRATEGY
        }
    
    # Get cache stats
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": DEVICE,
        "device_strategy": DEVICE_STRATEGY,
        "memory_info": memory_info,
        "cache_entries": len(cache_files)
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
    # Use localhost if behind a reverse proxy like Nginx
    host = "127.0.0.1" if os.path.exists("/etc/nginx/sites-enabled/deepseek") else "0.0.0.0"
    uvicorn.run("app:app", host=host, port=8000, reload=False)