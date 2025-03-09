import torch
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from .memory import clean_memory, get_memory_usage
from .device import DEVICE

logger = logging.getLogger("deepseek-api")

# Constants
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.6
TOP_P = 0.95

def load_model(model_name):
    """Load model and tokenizer separately for memory efficiency"""
    global_tokenizer = None
    
    # First load just the tokenizer to reduce peak memory
    logger.info("Loading tokenizer only first...")
    global_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully")
    
    # Log memory usage after tokenizer loading
    mem_info = get_memory_usage()
    logger.info(f"Memory after tokenizer: {mem_info['rss_mb']:.1f}MB, {mem_info['percent']:.1f}%")
    
    # Then load the model with memory optimizations
    logger.info(f"Loading model {model_name}...")
    
    try:
        # Memory optimization settings
        logger.info("Loading model with memory optimizations...")
        
        # In load_model function, update the model loading parameters:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,   # Try half precision instead of float32
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_state_dict=True,     # Add this to offload state dict during loading
            offload_folder="offload",
            device_map="cpu"
        )
        
        # Ensure model is on CPU by iterating through all parameters
        device_types = set()
        for param in model.parameters():
            device_types.add(param.device.type)
            if param.device.type != "cpu":
                logger.warning(f"Found parameter on {param.device}, moving to CPU")
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

I will extract only the sections, symptoms, diagnoses, treatments, lab results, and other information that are directly relevant to answering the specific question. I will ignore irrelevant information.
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
            # Very conservative max tokens
            actual_max_tokens = min(max_length, 300)
            
            # Generate with very conservative settings
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
            del input_ids
            del attention_mask
            if 'output' in locals():
                del output
            clean_memory()