from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import sys
import logging
import time
import torch
import psutil
import uvicorn
from contextlib import asynccontextmanager

# Force single-process behavior
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

# Run resource cleanup at exit
def cleanup_resources():
    """Clean up resources explicitly"""
    # Force garbage collection
    import gc
    gc.collect()
    
    # Close any open file handles
    for fd in range(3, 1000):
        try:
            os.close(fd)
        except OSError:
            pass
    
    logger.info("Resources cleaned up on exit")

# Register the cleanup handler
import atexit
atexit.register(cleanup_resources)

# App lifespan for model loading/unloading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up on startup
    clean_memory()
    time.sleep(1)  # Add a short delay
    
    # Make sure MPS is disabled
    disable_mps()
    time.sleep(1)  # Add a short delay
    
    # Load model and tokenizer
    app.state.model, app.state.tokenizer = load_model(MODEL_NAME)
    
    yield
    
    # Clean up on shutdown
    logger.info("Shutting down and cleaning up...")
    if hasattr(app.state, 'model'):
        del app.state.model
    if hasattr(app.state, 'tokenizer'):
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
    processing_time_seconds: Optional[float] = Field(None, description="Processing time in seconds")

@app.post("/process-context", response_model=ContextResponse)
async def process_context(request: ContextRequest):
    """Process patient data and extract relevant context for a specific query."""
    start_time = time.time()
    
    try:
        # Log memory at start of request
        mem_start = get_memory_usage()
        logger.info(f"Request started - Memory: {mem_start['rss_mb']:.1f}MB, {mem_start['percent']:.1f}%")
        
        # Record timing for each phase
        phase_timings = {}
        
        # Check cache first
        cache_start = time.time()
        cached_result = get_cached_result(
            request.patient_data, 
            request.query, 
            request.max_context_length
        )
        phase_timings["cache_check"] = time.time() - cache_start
        
        if cached_result:
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Returning cached result - Total time: {total_time:.2f} seconds")
            return ContextResponse(
                relevant_context=cached_result["context"],
                cached=True
            )
        
        # Use the DeepSeek model
        logger.info(f"Processing request with query: {request.query[:50]}...")
        
        # Measure model inference time
        inference_start = time.time()
        result = extract_context(
            app.state.model,
            app.state.tokenizer,
            request.patient_data,
            request.query,
            max_length=request.max_context_length
        )
        phase_timings["model_inference"] = time.time() - inference_start
        
        # Save to cache
        cache_save_start = time.time()
        save_to_cache(
            request.patient_data, 
            request.query, 
            request.max_context_length, 
            result
        )
        phase_timings["cache_save"] = time.time() - cache_save_start
        
        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log memory and timing information
        mem_end = get_memory_usage()
        logger.info(f"Request completed - Memory: {mem_end['rss_mb']:.1f}MB, {mem_end['percent']:.1f}%")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Phase timings: Cache check: {phase_timings['cache_check']:.2f}s, " + 
                   f"Model inference: {phase_timings['model_inference']:.2f}s, " + 
                   f"Cache save: {phase_timings['cache_save']:.2f}s")
        
        # Update the ContextResponse class to include processing_time
        class ContextResponseWithTiming(ContextResponse):
            processing_time_seconds: float
        
        return ContextResponseWithTiming(
            relevant_context=result["context"],
            cached=False,
            processing_time_seconds=total_time
        )
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error after {error_time:.2f} seconds: {str(e)}", exc_info=True)
        clean_memory()  # Force cleanup on error
        raise HTTPException(status_code=500, detail=f"Error processing context: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    # Get cache stats
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    # Get memory info
    mem_info = get_memory_usage()
    
    # Get system memory info
    system_mem = psutil.virtual_memory()
    
    # Get multiprocessing info
    mp_info = {}
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        mp_info = {
            "child_processes": len(children),
            "open_files": len(current_process.open_files()) if hasattr(current_process, 'open_files') else -1,
            "connections": len(current_process.connections()) if hasattr(current_process, 'connections') else -1,
        }
    except:
        mp_info = {"error": "Could not get multiprocessing info"}
    
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
        "multiprocessing": mp_info,
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
        # Run with explicit single worker to avoid multiprocessing
        uvicorn.run(
            "app:app", 
            host=host, 
            port=port, 
            log_level="info", 
            reload=False,
            workers=1,  # Enforce single process
            loop="asyncio"
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.error(f"Port {port} is still in use despite termination attempts")
            sys.exit(1)
        else:
            raise