import os
import signal
import subprocess
import time
import logging

logger = logging.getLogger("deepseek-api")

# Function to terminate any process using port 8000
def kill_process_on_port(port=8000):
    """Find and kill any process using the specified port"""
    try:
        # For macOS
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            pids = result.stdout.strip().split("\n")
            logger.info(f"Found processes using port {port}: {pids}")
            
            for pid in pids:
                try:
                    pid = int(pid.strip())
                    # Don't kill our own process
                    if pid != os.getpid():
                        logger.info(f"Killing process {pid} using port {port}")
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.5)  # Give it a moment to terminate
                        # Check if it's still running and force kill if needed
                        try:
                            os.kill(pid, 0)  # This will raise an error if process is gone
                            logger.info(f"Process {pid} still alive, sending SIGKILL")
                            os.kill(pid, signal.SIGKILL)
                        except OSError:
                            logger.info(f"Process {pid} terminated successfully")
                except (ValueError, ProcessLookupError) as e:
                    logger.warning(f"Error processing PID {pid}: {str(e)}")
            
            # Verify port is now free
            time.sleep(1)  # Wait a bit for the OS to release the port
            check = subprocess.run(
                ["lsof", "-i", f":{port}", "-t"], 
                capture_output=True, 
                text=True
            )
            if check.stdout.strip():
                logger.warning(f"Port {port} still in use after termination attempts")
            else:
                logger.info(f"Port {port} is now free")
        else:
            logger.info(f"No process found using port {port}")
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {str(e)}")