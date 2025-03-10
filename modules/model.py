import torch
import json
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from .memory import clean_memory, get_memory_usage
from .device import DEVICE

logger = logging.getLogger("deepseek-api")

# Disable multiprocessing for PyTorch to avoid semaphore leaks
os.environ["OMP_NUM_THREADS"] = "1"  # Force single-threaded behavior
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads

# Constants
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.6
TOP_P = 0.95

def load_model(model_name):
    """Load model and tokenizer with minimal multiprocessing"""
    global_tokenizer = None
    
    # First load just the tokenizer
    logger.info("Loading tokenizer...")
    global_tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    logger.info("Tokenizer loaded successfully")
    
    # Log memory usage after tokenizer loading
    mem_info = get_memory_usage()
    logger.info(f"Memory after tokenizer: {mem_info['rss_mb']:.1f}MB, {mem_info['percent']:.1f}%")
    
    # Clean memory before model loading
    clean_memory()
    
    # Then load the model with minimal multiprocessing
    logger.info(f"Loading model {model_name} with single-process approach...")
    
    try:
        # Minimal options to avoid multiprocessing
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,      # Use float16 for memory efficiency
            trust_remote_code=True,
            low_cpu_mem_usage=False,        # Avoid this to reduce multiprocessing
            offload_state_dict=False,       # Avoid this to reduce multiprocessing
            local_files_only=False,         # Set to True if model is already downloaded
            device_map=None                 # No device mapping, just load directly to device
        )
        
        # Move model to CPU explicitly after loading
        model = model.to(DEVICE)
        
        # Verify device location
        device_types = set()
        for name, param in model.named_parameters():
            device_types.add(param.device.type)
            if param.device.type != "cpu":
                logger.warning(f"Parameter {name} on {param.device}, moving to CPU")
                param.data = param.data.to("cpu")
        
        logger.info(f"Model parameters are on devices: {device_types}")
        logger.info("Model loaded successfully on CPU!")
        
        # Log memory after model loading
        mem_info = get_memory_usage()
        logger.info(f"Memory after model loading: {mem_info['rss_mb']:.1f}MB, {mem_info['percent']:.1f}%")
        
        return model, global_tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Simplified patient data processing
def process_large_patient_data(patient_data):
    """Break down large patient data to fit in context window"""
    data_str = json.dumps(patient_data)
    
    # If data is small enough, return as is
    if len(data_str) < 20000:  # Very conservative 20KB limit
        return patient_data
    
    # Otherwise truncate
    logger.warning("Patient data is very large, truncating to fit in context window")
    truncated_str = data_str[:20000] + "... [truncated due to size]"
    
    try:
        return json.loads(truncated_str)
    except:
        return {"data": "Patient data was too large and was truncated"}

def extract_context(model, tokenizer, patient_data, query, max_length=300):
    """Use the DeepSeek model to extract relevant context from patient data."""
    # Process large data
    processed_patient_data = process_large_patient_data(patient_data)
    
    # Format the input
    prompt = f"""<think>
I need to analyze a patient's medical record to extract only the information relevant to the following question:
"{query}"

Here is the complete patient record:
{json.dumps(processed_patient_data, indent=2)}

I will extract only the health data and information that are directly relevant to answering the specific question. I will ignore irrelevant information.
</think>

Based on the medical record, here is the relevant information to answer the question:
"""
    
    # Check memory before generation
    mem_before = get_memory_usage()
    logger.info(f"Memory before generation: {mem_before['rss_mb']:.1f}MB, {mem_before['percent']:.1f}%")
    
    # Generate response 
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    
    logger.info(f"Input shape: {input_ids.shape}, device: {input_ids.device}")
    
    with torch.no_grad():
        clean_memory()
        
        try:
            # Use a safer and smaller max token count
            actual_max_tokens = min(max_length, 200)  # Even more conservative
            
            # Simple, single-process generation
            logger.info(f"Starting generation with max_new_tokens={actual_max_tokens}")
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=actual_max_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV caching for efficiency
            )
            
            # Decode the output
            full_output = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            relevant_context = full_output[len(prompt):].strip()
            
            # Check memory after generation
            mem_after = get_memory_usage()
            logger.info(f"Memory after generation: {mem_after['rss_mb']:.1f}MB, {mem_after['percent']:.1f}%")
            
            return {"context": relevant_context}
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            # Try a simplified response if generation fails
            return {"context": f"Error processing patient data: {str(e)}. Please try with a simpler query or less data."}
        finally:
            # Force cleanup of generation tensors
            if 'input_ids' in locals():
                del input_ids
            if 'attention_mask' in locals():
                del attention_mask
            if 'output' in locals():
                del output
            clean_memory()