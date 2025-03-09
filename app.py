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
import logging
import time
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

# Force CPU mode for stability
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_ENABLE_FALLBACK"] = "0"  # Disable MPS fallback
DEVICE = "cpu"

# Set reasonable thread count for CPU
NUM_THREADS = max(1, min(os.cpu_count() - 1, 4))  # Cap at 4 threads to avoid overuse
torch.set_num_threads(NUM_THREADS)
logger.info(f"Set PyTorch to use {NUM_THREADS} CPU threads")

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
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# App lifespan for model loading/unloading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer on startup
    logger.info(f"Loading model {MODEL_NAME}...")
    clean_memory()
    
    # Download tokenizer
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully")
    
    # Load model with minimal parameters and in simple CPU mode
    logger.info("Loading model in CPU mode...")
    try:
        app.state.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,      # Use full precision for stability
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None                # No device map - simple loading
        )
        
        # Ensure model is on CPU
        app.state.model = app.state.model.to(DEVICE)
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
    
    # Generate response
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        clean_memory()
        
        try:
            # Limit generation length for safety
            actual_max_tokens = min(max_length, 500)
            
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
    
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": DEVICE,
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
    # Use localhost binding
    host = "127.0.0.1" 
    uvicorn.run("app:app", host=host, port=8000, log_level="info", reload=False)