from fastapi import FastAPI, HTTPException, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

# Force CPU mode with quantization to ensure stability
DEVICE = "cpu"
logger.info("Using CPU for inference with 8-bit quantization")

# Optimize CPU threads for performance
torch.set_num_threads(max(1, os.cpu_count() - 1))
logger.info(f"Set PyTorch to use {torch.get_num_threads()} CPU threads")

# Performance optimization settings
CACHE_DIR = os.path.join(os.path.expanduser("~"), "deepseek-app", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
    
    # Clean up before loading
    clean_memory()
    
    logger.info(f"Downloading model from Hugging Face: {MODEL_NAME}")
    # Ensure the directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Download the tokenizer
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully")
    
    # Configure 8-bit quantization
    logger.info("Configuring 8-bit quantization for reduced memory usage")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    # Load model with 8-bit quantization
    logger.info("Loading model with 8-bit quantization...")
    try:
        app.state.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded successfully with 8-bit quantization!")
    except Exception as e:
        # Fallback to 4-bit if 8-bit fails
        logger.warning(f"8-bit quantization failed: {str(e)}. Trying 4-bit...")
        clean_memory()
        
        # 4-bit config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        try:
            app.state.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )
            logger.info("Model loaded successfully with 4-bit quantization!")
        except Exception as e2:
            # Last resort: try to load a smaller model
            logger.error(f"4-bit quantization also failed: {str(e2)}. Your system may not have enough resources.")
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
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Create an attention mask to avoid warnings
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        # Clean memory before generation
        clean_memory()
        
        try:
            # Set smaller max_new_tokens if input is large
            actual_max_tokens = min(max_length, 800)  # Limit to prevent memory issues
            
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
    
    # Get memory info
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
    
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": DEVICE,
        "quantization": "8-bit",
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
    uvicorn.run("app:app", host=host, port=8000, log_level="info", reload=False)