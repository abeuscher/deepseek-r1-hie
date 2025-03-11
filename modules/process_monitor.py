import os
import psutil
import logging
import time
import signal

logger = logging.getLogger("deepseek-api.monitor")

def find_stale_processes(port=8000):
    """Find any stale processes that might be using our port or resources."""
    stale_processes = []
    
    # Check for processes using our port
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's using our port
            for conn in proc.connections():
                if conn.laddr.port == port:
                    stale_processes.append(proc)
                    break
                    
            # Check if it's a Python/uvicorn process with our app name
            if proc.info['cmdline'] and 'python' in ' '.join(proc.info['cmdline']).lower():
                if 'app.py' in ' '.join(proc.info['cmdline']):
                    if proc.pid != os.getpid():  # Don't include our own process
                        stale_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return stale_processes

def terminate_stale_processes(port=8000):
    """Terminate any stale processes that might interfere with service startup."""
    stale_procs = find_stale_processes(port)
    
    if stale_procs:
        logger.warning(f"Found {len(stale_procs)} stale processes that may interfere with service")
        for proc in stale_procs:
            try:
                logger.warning(f"Terminating stale process: PID {proc.pid}, Name: {proc.name()}")
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.error(f"Failed to terminate process {proc.pid}")
        
        # Give them time to terminate gracefully
        time.sleep(2)
        
        # Check if any are still alive and force kill
        for proc in stale_procs:
            try:
                if proc.is_running():
                    logger.warning(f"Force killing process: {proc.pid}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return len(stale_procs)
    
    return 0
