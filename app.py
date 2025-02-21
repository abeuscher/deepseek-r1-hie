from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95

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
    app.state.model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",  # Let the library decide the optimal mapping
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Add this for more memory efficiency
        offload_folder="offload"  # Enable disk offloading if needed
    )
    logger.info("Model loaded successfully!")
    yield
    # Clean up on shutdown
    logger.info("Shutting down and cleaning up...")
    del app.state.model
    del app.state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="DeepSeek-R1 Reasoning API", lifespan=lifespan)

class ContextRequest(BaseModel):
    """Request model for context processing."""
    patient_data: Dict[str, Any] = Field(..., description="The full patient record data")
    query: str = Field(..., description="The user's query about the patient")
    max_context_length: Optional[int] = Field(1000, description="Maximum number of tokens for the returned context")

class ContextResponse(BaseModel):
    """Response model with filtered context."""
    relevant_context: str = Field(..., description="The extracted relevant context from the patient data")
    reasoning: Optional[str] = Field(None, description="Optional explanation of the reasoning process")

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
    
    with torch.no_grad():
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
async def process_context(request: ContextRequest):
    """
    Process patient data and extract relevant context for a specific query.
    """
    try:
        # Use the DeepSeek model to extract relevant context
        result = extract_context(
            app.state.model,
            app.state.tokenizer,
            request.patient_data,
            request.query,
            max_length=request.max_context_length
        )
        
        return ContextResponse(
            relevant_context=result["context"],
            reasoning=result.get("reasoning")
        )
    except Exception as e:
        logger.error(f"Error processing context: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing context: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME, "device": DEVICE}

if __name__ == "__main__":
    # Use localhost if behind a reverse proxy like Nginx
    host = "127.0.0.1" if os.path.exists("/etc/nginx/sites-enabled/deepseek") else "0.0.0.0"
    uvicorn.run("app:app", host=host, port=8000, reload=False)