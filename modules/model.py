import torch
import json
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from .memory import clean_memory, get_memory_usage, MemoryCheckpoint, check_memory_threshold
from .device import DEVICE

logger = logging.getLogger("deepseek-api")

# Constants
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.6
TOP_P = 0.95

# Generation safety - max tokens per batch
MAX_TOKENS_PER_BATCH = 50  # Only generate this many tokens at a time

def load_model(model_name):
    """Load model and tokenizer separately for memory efficiency"""
    global_tokenizer = None
    
    # First load just the tokenizer to reduce peak memory
    with MemoryCheckpoint("tokenizer_loading"):
        logger.info("Loading tokenizer only first...")
        global_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
    
    # Log memory usage after tokenizer loading
    mem_info = get_memory_usage()
    logger.info(f"Memory after tokenizer: {mem_info['rss_mb']:.1f}MB, {mem_info['percent']:.1f}%")
    
    # Clean memory before model loading
    clean_memory()
    
    # Then load the model with memory optimizations
    with MemoryCheckpoint("model_loading"):
        logger.info(f"Loading model {model_name}...")
        
        try:
            # Memory optimization settings
            logger.info("Loading model with memory optimizations...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,   # Changed to float16 for memory efficiency
                trust_remote_code=True,
                low_cpu_mem_usage=True,      # Enable low memory usage
                offload_state_dict=True,     # Added for memory efficiency
                offload_folder="offload",    # Setup offload folder for memory efficiency
                device_map="cpu"             # Explicitly set to CPU
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
    
    # Check if we can safely proceed or if memory is already high
    if not check_memory_threshold():
        logger.error("Memory usage too high before starting generation, aborting")
        return {"context": "Error: System memory usage too high to safely process the request. Please try again later."}
    
    # Process large data
    with MemoryCheckpoint("process_patient_data"):
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
    
    # Track major steps with memory checkpoints
    with MemoryCheckpoint("tokenize_input"):
        # Generate response 
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        attention_mask = torch.ones_like(input_ids)
        logger.info(f"Input shape: {input_ids.shape}, device: {input_ids.device}")
    
    # Generate with careful memory monitoring
    with torch.no_grad():
        clean_memory()
        
        try:
            # Set very conservative max tokens
            actual_max_tokens = min(max_length, 300)
            
            # Use chunked generation for safety - generate tokens in batches
            if MAX_TOKENS_PER_BATCH < actual_max_tokens:
                logger.info(f"Using chunked generation with {MAX_TOKENS_PER_BATCH} tokens per batch")
                
                with MemoryCheckpoint("generation_setup"):
                    # Prepare for batched generation
                    current_input_ids = input_ids
                    all_generated_ids = None
                    tokens_generated = 0
                    
                # Generate tokens in batches to control memory usage
                while tokens_generated < actual_max_tokens:
                    # Decide how many tokens to generate in this batch
                    tokens_to_generate = min(MAX_TOKENS_PER_BATCH, actual_max_tokens - tokens_generated)
                    
                    # Check memory before generating batch
                    if not check_memory_threshold():
                        logger.error(f"Memory threshold exceeded after generating {tokens_generated} tokens. Stopping generation.")
                        break
                    
                    # Generate batch
                    batch_checkpoint_name = f"generate_batch_{tokens_generated}_{tokens_generated+tokens_to_generate}"
                    with MemoryCheckpoint(batch_checkpoint_name):
                        logger.info(f"Generating batch of {tokens_to_generate} tokens")
                        batch_output = model.generate(
                            current_input_ids,
                            attention_mask=torch.ones_like(current_input_ids),
                            max_new_tokens=tokens_to_generate,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Update tracking
                    if all_generated_ids is None:
                        all_generated_ids = batch_output
                    else:
                        # Use the newly generated part to continue
                        continuation = batch_output[:, current_input_ids.shape[1]:]
                        all_generated_ids = torch.cat([all_generated_ids, continuation], dim=1)
                    
                    # Update for next iteration
                    current_input_ids = batch_output
                    tokens_generated += tokens_to_generate
                    
                    # Check if generation is complete before max tokens
                    if batch_output[0, -1].item() == tokenizer.eos_token_id:
                        logger.info("Generation complete - EOS token reached")
                        break
                    
                    # Clean memory between batches
                    clean_memory()
                    
                # Use the final output
                output = all_generated_ids
                
            else:
                # For very small generations, do it all at once
                with MemoryCheckpoint("model_generate"):
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
            
            # Decoding phase
            with MemoryCheckpoint("decode_output"):
                # Decode the output
                full_output = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the generated part (after the prompt)
                relevant_context = full_output[len(prompt):].strip()
            
            # Check memory after generation
            mem_after = get_memory_usage()
            logger.info(f"Memory after generation: {mem_after['rss_mb']:.1f}MB, {mem_after['percent']:.1f}%")
            logger.info(f"Memory change during generation: {mem_after['rss_mb'] - mem_before['rss_mb']:.1f}MB")
            
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
            if 'current_input_ids' in locals():
                del current_input_ids
            if 'batch_output' in locals():
                del batch_output
            if 'all_generated_ids' in locals():
                del all_generated_ids
            clean_memory()