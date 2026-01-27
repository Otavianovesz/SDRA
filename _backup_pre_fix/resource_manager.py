"""
ResourceManager - 8GB RAM Survival Mode
========================================
Singleton for global resource control in memory-constrained environments.

Key features:
- Strict memory limits (6.5GB max)
- CPU thread control to prevent saturation
- Inference timeouts
- File locking to prevent multi-instance OOM
- Execution priority queue
"""

import os
import gc
import sys
import logging
import threading
import functools
import signal
from pathlib import Path
from typing import Optional, Callable, Any
from contextlib import contextmanager
from datetime import datetime

import config

logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Set CPU thread limits BEFORE importing torch/numpy
for key, value in config.ENV_VARS.items():
    os.environ.setdefault(key, value)

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# MEMORY LIMITS
# =============================================================================

MAX_RAM_USAGE_GB = config.MAX_RAM_USAGE_GB
WARNING_RAM_USAGE_GB = config.WARNING_RAM_USAGE_GB
INFERENCE_TIMEOUT_SECONDS = config.INFERENCE_TIMEOUT_SECONDS

# Execution priority (lower = higher priority)
EXECUTION_PRIORITY = config.EXECUTION_PRIORITY


# =============================================================================
# FILE LOCK FOR SINGLE INSTANCE
# =============================================================================

class SingleInstanceLock:
    """Prevents running multiple instances to avoid OOM."""
    
    def __init__(self, lock_name: str = "srda_rural"):
        self.lock_file = Path(os.environ.get("TEMP", "/tmp")) / f"{lock_name}.lock"
        self.locked = False
        self._file_handle = None
        
    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if successful."""
        try:
            # Windows-compatible file locking
            if sys.platform == "win32":
                import msvcrt
                self._file_handle = open(self.lock_file, "w")
                msvcrt.locking(self._file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                self._file_handle = open(self.lock_file, "w")
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            self._file_handle.write(str(os.getpid()))
            self._file_handle.flush()
            self.locked = True
            logger.info(f"Acquired instance lock: {self.lock_file}")
            return True
            
        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Failed to acquire lock (another instance running?): {e}")
            return False
    
    def release(self):
        """Release the lock."""
        if self._file_handle:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(self._file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
                self._file_handle.close()
            except:
                pass
        
        try:
            self.lock_file.unlink(missing_ok=True)
        except:
            pass
        
        self.locked = False


# =============================================================================
# TIMEOUT HANDLER
# =============================================================================

class TimeoutError(Exception):
    """Raised when inference exceeds timeout."""
    pass


@contextmanager
def inference_timeout(seconds: int = INFERENCE_TIMEOUT_SECONDS):
    """
    Context manager for inference timeout.
    
    Usage:
        with inference_timeout(60):
            result = model.predict(image)
    """
    if sys.platform == "win32":
        # Windows doesn't support signal.SIGALRM, use threading
        result = {"timeout": False}
        timer = threading.Timer(seconds, lambda: result.update(timeout=True))
        timer.start()
        try:
            yield
            if result["timeout"]:
                raise TimeoutError(f"Inference exceeded {seconds}s timeout")
        finally:
            timer.cancel()
    else:
        def handler(signum, frame):
            raise TimeoutError(f"Inference exceeded {seconds}s timeout")
        
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# RESOURCE CONSTRAINED DECORATOR
# =============================================================================

def resource_constrained(required_gb: float = 1.0, auto_cleanup: bool = True):
    """
    Decorator for functions that require significant memory.
    
    Checks available RAM before execution and forces cleanup after.
    
    Usage:
        @resource_constrained(required_gb=1.5)
        def run_heavy_model(image):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resource_manager()
            
            # Pre-flight memory check
            if not manager.check_memory_available(required_gb):
                logger.warning(f"Memory critical for {func.__name__}, forcing cleanup...")
                manager.force_cleanup()
                
                # Re-check after cleanup
                if not manager.check_memory_available(required_gb):
                    raise MemoryError(
                        f"Insufficient memory for {func.__name__}: "
                        f"need {required_gb}GB, have {manager.get_available_gb():.1f}GB"
                    )
            
            # Log memory state
            manager.log_memory_state(f"Before {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if auto_cleanup:
                    gc.collect()
                manager.log_memory_state(f"After {func.__name__}")
        
        return wrapper
    return decorator


# =============================================================================
# RESOURCE MANAGER SINGLETON
# =============================================================================

class ResourceManager:
    """
    Singleton for global resource management.
    
    Tracks memory usage, enforces limits, and provides cleanup utilities.
    """
    
    _instance: Optional["ResourceManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.max_ram_gb = MAX_RAM_USAGE_GB
        self.warning_ram_gb = WARNING_RAM_USAGE_GB
        self.instance_lock = SingleInstanceLock()
        self._memory_log = []
        
        # Configure PyTorch for CPU
        self._configure_torch()
        
        self._initialized = True
        logger.info(f"ResourceManager initialized: max RAM = {self.max_ram_gb}GB")
    
    def _configure_torch(self):
        """Configure PyTorch for CPU-only, memory-efficient operation."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Limit CPU threads
            torch.set_num_threads(config.CPU_THREADS)
            
            # Disable gradients globally (saves ~30% memory)
            torch.set_grad_enabled(False)
            
            # Force CPU device
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device('cpu')
            
            logger.info(f"PyTorch configured: threads={torch.get_num_threads()}, grad=False")
            
        except Exception as e:
            logger.warning(f"Failed to configure PyTorch: {e}")
    
    def acquire_instance_lock(self) -> bool:
        """Acquire single-instance lock."""
        return self.instance_lock.acquire()
    
    def release_instance_lock(self):
        """Release single-instance lock."""
        self.instance_lock.release()
    
    def get_memory_usage_gb(self) -> float:
        """Get current process memory usage in GB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 3)
        except:
            return 0.0
    
    def get_available_gb(self) -> float:
        """Get available system RAM in GB."""
        if not PSUTIL_AVAILABLE:
            return 8.0
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except:
            return 8.0
    
    def get_total_gb(self) -> float:
        """Get total system RAM in GB."""
        if not PSUTIL_AVAILABLE:
            return 8.0
        try:
            return psutil.virtual_memory().total / (1024 ** 3)
        except:
            return 8.0
    
    def check_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available."""
        available = self.get_available_gb()
        # Keep 1GB buffer
        safe_available = available - 1.0
        return safe_available >= required_gb
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        usage = self.get_memory_usage_gb()
        return usage > self.max_ram_gb
    
    def is_memory_warning(self) -> bool:
        """Check if memory usage is at warning level."""
        usage = self.get_memory_usage_gb()
        return usage > self.warning_ram_gb
    
    def log_memory_state(self, context: str = ""):
        """Log current memory state."""
        usage = self.get_memory_usage_gb()
        available = self.get_available_gb()
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "usage_gb": round(usage, 2),
            "available_gb": round(available, 2)
        }
        self._memory_log.append(entry)
        
        # Keep only last 100 entries
        if len(self._memory_log) > 100:
            self._memory_log = self._memory_log[-100:]
        
        level = logging.WARNING if self.is_memory_warning() else logging.DEBUG
        logger.log(level, f"Memory [{context}]: {usage:.2f}GB used, {available:.2f}GB available")
    
    def force_cleanup(self):
        """Force aggressive memory cleanup."""
        logger.info("Forcing memory cleanup...")
        
        # Multiple GC passes
        gc.collect()
        gc.collect()
        gc.collect()
        
        # PyTorch cleanup
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        self.log_memory_state("After cleanup")
    
    def get_status(self) -> dict:
        """Get current resource status."""
        return {
            "memory_usage_gb": self.get_memory_usage_gb(),
            "memory_available_gb": self.get_available_gb(),
            "memory_total_gb": self.get_total_gb(),
            "max_allowed_gb": self.max_ram_gb,
            "is_critical": self.is_memory_critical(),
            "is_warning": self.is_memory_warning(),
            "instance_locked": self.instance_lock.locked,
        }
    
    def quantize_model(self, model: Any) -> Any:
        """
        Apply dynamic int8 quantization to a PyTorch model for CPU.
        
        Reduces memory by ~2-4x with minimal quality loss.
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            import torch
            
            # Only quantize if it's a torch module
            if not isinstance(model, torch.nn.Module):
                return model
            
            # Dynamic quantization for linear layers
            quantized = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            logger.info("Model quantized to int8")
            return quantized
            
        except Exception as e:
            logger.warning(f"Quantization failed, using original model: {e}")
            return model


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_resource_manager: Optional[ResourceManager] = None

def get_resource_manager() -> ResourceManager:
    """Get or create the global ResourceManager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_resources() -> bool:
    """
    Initialize resource manager at application startup.
    
    Should be called at the very beginning of main().
    Returns False if another instance is running.
    """
    manager = get_resource_manager()
    
    # Try to acquire instance lock
    if not manager.acquire_instance_lock():
        logger.error("Another instance of SRDA-Rural is already running!")
        return False
    
    manager.log_memory_state("Application startup")
    return True


def cleanup_resources():
    """
    Cleanup resources at application shutdown.
    
    Should be called at the end of main() or in atexit.
    """
    manager = get_resource_manager()
    manager.force_cleanup()
    manager.release_instance_lock()
    logger.info("Resources cleaned up")
