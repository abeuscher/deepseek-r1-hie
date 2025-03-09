import os
import torch
import logging

logger = logging.getLogger("deepseek-api")

# CRITICAL: Completely disable GPU and MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_ENABLE_FALLBACK"] = "0"

# Explicitly disable MPS availability detection
def disable_mps():
    if hasattr(torch.backends, "mps"):
        # Use a direct monkeypatch approach to ensure MPS is never used
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
        # Set these for older PyTorch versions
        if hasattr(torch, "has_mps"):
            torch.has_mps = False
    logger.info("MPS (Metal Performance Shaders) forcibly disabled")

# Call the function to disable MPS
disable_mps()

# Set device to CPU explicitly
DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

# Set reasonable thread count for CPU
NUM_THREADS = max(1, min(os.cpu_count() - 1, 4))  # Cap at 4 threads to avoid overuse
torch.set_num_threads(NUM_THREADS)
logger.info(f"Set PyTorch to use {NUM_THREADS} CPU threads")