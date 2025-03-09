import os
import gc
import torch
import psutil
import logging

logger = logging.getLogger("deepseek-api")

# Function to get memory usage
def get_memory_usage():
    """Get current memory usage of this process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Convert to MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Convert to MB
        "percent": process.memory_percent()
    }

# Memory management
def clean_memory():
    """Force garbage collection"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Log memory usage
    mem_info = get_memory_usage()
    logger.info(f"Memory cleaned - Current usage: {mem_info['rss_mb']:.1f}MB, {mem_info['percent']:.1f}%")