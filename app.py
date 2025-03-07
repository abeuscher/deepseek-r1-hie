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

# Performance optimization settings
CACHE_DIR = os.path.join(os.path.expanduser("~"), "deepseek-app", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure device with Apple Silicon optimization
if torch.backends.mps.is_available():
    DEVICE = "mps"
    # Enable Metal Performance Shaders optimizations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logger.info("Using Apple Silicon GPU (MPS) for inference")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    # Enable TF32 for better performance on NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
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
    
    model_kwargs = {
        "pretrained_model_name_or_path": MODEL_NAME,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "offload_folder": "offload"
    }
    
    # Device-specific optimizations
    if DEVICE == "cuda":
        model_kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.float16
        })
    elif DEVICE == "mps":
        # For Apple Silicon, using bfloat16 if available, otherwise float16
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
    
    # Make sure input is on the same device as the model
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        # Set autocast for MPS/CUDA to improve performance
        if DEVICE == "mps" or DEVICE == "cuda":
            with torch.autocast(device_type=DEVICE):
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True
                )
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
    
    # Get cache stats
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": DEVICE,
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