from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import sys
import logging
import time
from contextlib import asynccontextmanager

# Import our modules
from modules.memory import clean_memory, get_memory_usage
from modules.device import DEVICE, disable_mps, NUM_THREADS
from modules.cache import get_cached_result, save_to_cache, CACHE_DIR
from modules.port import kill_process_on_port
from modules.model import load_model, extract_context

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("deepseek-api")

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# App lifespan for model loading/unloading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up on startup
    clean_memory()
    
    # Make sure MPS is disabled
    disable_mps()
    
    # Load model and tokenizer
    app.state.model, app.state.tokenizer = load_model(MODEL_NAME)
    
    yield
    
    # Clean up on shutdown
    logger.info("Shutting down and cleaning up...")
    del app.state.model
    del app.state.tokenizer
    app.state.model = None
    app.state.tokenizer = None
    clean_memory()
    logger.info("Model unloaded")

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
    max_context_length: Optional[int] = Field(300, description="Maximum number of tokens for the returned context")

class ContextResponse(BaseModel):
    """Response model with filtered context."""
    relevant_context: str = Field(..., description="The extracted relevant context from the patient data")
    cached: Optional[bool] = Field(None, description="Indicates if the result was retrieved from cache")

@app.post("/process-context", response_model=ContextResponse)
async def process_context(request: ContextRequest):
    """Process patient data and extract relevant context for a specific query."""
    try:
        # Log memory at start of request
        mem_start = get_memory_usage()
        logger.info(f"Request started - Memory: {mem_start['rss_mb']:.1f}MB, {mem_start['percent']:.1f}%")
        
        # Check cache first
        cached_result = get_cached_result(
            request.patient_data, 
            request.query, 
            request.max_context_length
        )
        
        if cached_result:
            logger.info("Returning cached result")
            return ContextResponse(
                relevant_context=cached_result["context"],
                cached=True
            )
        
        # Use the DeepSeek model
        logger.info(f"Processing request with query: {request.query[:50]}...")
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
        
        # Log memory at end of request
        mem_end = get_memory_usage()
        logger.info(f"Request completed - Memory: {mem_end['rss_mb']:.1f}MB, {mem_end['percent']:.1f}%")
        
        return ContextResponse(
            relevant_context=result["context"],
            cached=False
        )
    except Exception as e:
        logger.error(f"Error processing context: {str(e)}", exc_info=True)
        clean_memory()  # Force cleanup on error
        raise HTTPException(status_code=500, detail=f"Error processing context: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    import os
    import psutil
    import torch
    
    # Get cache stats
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    # Get memory info
    mem_info = get_memory_usage()
    
    # Get system memory info
    system_mem = psutil.virtual_memory()
    
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": DEVICE,
        "memory": {
            "process_mb": mem_info["rss_mb"],
            "process_percent": mem_info["percent"],
            "system_total_gb": system_mem.total / (1024**3),
            "system_available_gb": system_mem.available / (1024**3),
            "system_percent": system_mem.percent
        },
        "cpu_threads": torch.get_num_threads(),
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
        import uvicorn
        uvicorn.run("app:app", host=host, port=port, log_level="info", reload=False)
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.error(f"Port {port} is still in use despite termination attempts")
            sys.exit(1)
        else:
            raise