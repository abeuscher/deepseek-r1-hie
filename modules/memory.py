import os
import gc
import torch
import psutil
import logging
import time

logger = logging.getLogger("deepseek-api")

# Memory thresholds
MEMORY_WARN_THRESHOLD = 70  # Percentage of system memory that triggers a warning
MEMORY_ABORT_THRESHOLD = 80  # Percentage of system memory that triggers abortion

# Function to get memory usage
def get_memory_usage():
    """Get current memory usage of this process and system"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_mem = psutil.virtual_memory()
    
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Process resident memory in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Process virtual memory in MB
        "percent": process.memory_percent(),         # Process memory as % of system memory
        "system_percent": system_mem.percent,        # System memory usage percentage
        "system_available_mb": system_mem.available / (1024 * 1024),  # Available system memory in MB
        "system_total_mb": system_mem.total / (1024 * 1024)  # Total system memory in MB
    }

# Memory monitoring with detailed PyTorch tracking
def get_detailed_memory():
    """Get detailed memory usage including PyTorch allocations"""
    mem_stats = get_memory_usage()
    
    # Get PyTorch memory stats if available
    torch_mem = {}
    if hasattr(torch.cuda, 'memory_allocated'):
        torch_mem['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        torch_mem['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
    
    # Try to get MPS memory info if available (may not work on all systems)
    if hasattr(torch.backends, 'mps') and hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
        try:
            torch_mem['mps_allocated_mb'] = torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            torch_mem['mps_allocated_mb'] = 0
    
    # Combine stats
    mem_stats.update(torch_mem)
    return mem_stats

# Memory management
def clean_memory():
    """Force garbage collection"""
    before_mem = get_memory_usage()
    
    # Run garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Log memory usage
    after_mem = get_memory_usage()
    memory_freed = before_mem['rss_mb'] - after_mem['rss_mb']
    
    logger.info(f"Memory cleaned - Freed: {memory_freed:.1f}MB, Current: {after_mem['rss_mb']:.1f}MB ({after_mem['percent']:.1f}%)")
    logger.info(f"System memory: {after_mem['system_percent']:.1f}% used, {after_mem['system_available_mb']:.1f}MB available")

    return after_mem

# Memory monitoring with warning and abort thresholds
def check_memory_threshold():
    """Check if memory usage exceeds thresholds, returns True if safe, False if should abort"""
    mem_info = get_memory_usage()
    
    if mem_info['system_percent'] > MEMORY_ABORT_THRESHOLD:
        logger.error(f"CRITICAL: System memory usage ({mem_info['system_percent']:.1f}%) exceeds abort threshold ({MEMORY_ABORT_THRESHOLD}%)")
        logger.error(f"Process is using {mem_info['rss_mb']:.1f}MB ({mem_info['percent']:.1f}% of system memory)")
        return False
    
    if mem_info['system_percent'] > MEMORY_WARN_THRESHOLD:
        logger.warning(f"WARNING: High memory usage - System: {mem_info['system_percent']:.1f}%, Process: {mem_info['rss_mb']:.1f}MB")
    
    return True

# Memory checkpoint context manager
class MemoryCheckpoint:
    """Context manager to track memory usage at a checkpoint"""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.start_mem = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_mem = get_detailed_memory()
        logger.info(f"Memory checkpoint '{self.name}' started - Process: {self.start_mem['rss_mb']:.1f}MB, System: {self.start_mem['system_percent']:.1f}%")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_mem = get_detailed_memory()
        
        duration = end_time - self.start_time
        mem_change = end_mem['rss_mb'] - self.start_mem['rss_mb']
        
        logger.info(f"Memory checkpoint '{self.name}' ended - Duration: {duration:.2f}s, Memory change: {mem_change:.1f}MB")
        logger.info(f"Final memory state - Process: {end_mem['rss_mb']:.1f}MB, System: {end_mem['system_percent']:.1f}%")
        
        # Return False to abort if memory threshold exceeded
        return check_memory_threshold()