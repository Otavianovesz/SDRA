"""
LazyModelManager - Memory-Optimized Model Loading

Implements aggressive memory management for 8GB RAM constraint.
Models are loaded on-demand and unloaded immediately after use.
"""

import gc
import logging
from typing import Optional, Any, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")


class LazyModelManager:
    """
    Manages heavy AI models with strict memory discipline.
    
    Key principles:
    1. Load-Infer-Unload pattern for VLM
    2. Only one heavy model active at a time
    3. Aggressive gc.collect() after unload
    4. 7.5GB RAM threshold enforced
    """
    
    def __init__(self, memory_limit_gb: float = 7.5):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self._active_models: Dict[str, Any] = {}
        self._model_sizes = {
            "gliner": 1.2,  # ~1.2GB RAM
            "vlm": 2.5,     # ~2.5GB RAM (SmolDocling)
            "qwen": 4.0,    # ~4GB RAM (Qwen2-VL) - DO NOT USE with 8GB
        }
    
    def get_memory_usage_gb(self) -> float:
        """Get current RAM usage in GB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        return psutil.virtual_memory().used / (1024 ** 3)
    
    def get_available_memory_gb(self) -> float:
        """Get available RAM in GB."""
        if not PSUTIL_AVAILABLE:
            return 8.0  # Assume 8GB if can't check
        return psutil.virtual_memory().available / (1024 ** 3)
    
    def check_memory(self, required_gb: float = 1.0) -> bool:
        """
        Check if there's enough RAM for a new model.
        
        Args:
            required_gb: Estimated RAM needed for the model
            
        Returns:
            True if safe to load, False if would exceed limit
        """
        if not PSUTIL_AVAILABLE:
            return True  # Proceed without check
        
        mem = psutil.virtual_memory()
        current_used = mem.used
        would_use = current_used + (required_gb * 1024 ** 3)
        
        if would_use > self.memory_limit_bytes:
            logger.critical(
                f"MEMORY LIMIT: Would use {would_use/1e9:.2f}GB, "
                f"limit is {self.memory_limit_bytes/1e9:.2f}GB"
            )
            return False
        
        logger.debug(f"Memory check OK: {current_used/1e9:.2f}GB used, "
                     f"need {required_gb}GB, limit {self.memory_limit_bytes/1e9:.2f}GB")
        return True
    
    def unload_all(self):
        """Aggressively unload all models and free memory."""
        if not self._active_models:
            return
        
        model_names = list(self._active_models.keys())
        logger.info(f"Unloading models: {model_names}")
        
        for name in model_names:
            try:
                model = self._active_models.pop(name)
                del model
            except Exception as e:
                logger.warning(f"Error unloading {name}: {e}")
        
        self._active_models = {}
        
        # Aggressive garbage collection
        gc.collect()
        gc.collect()  # Second pass for cyclic references
        
        # Try to release GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Memory cleanup complete")
    
    def load_gliner(self) -> Optional[Any]:
        """
        Load GLiNER model with memory safety.
        
        Returns:
            GLiNER model or None if memory insufficient
        """
        if "gliner" in self._active_models:
            return self._active_models["gliner"]
        
        # Unload other heavy models first
        self.unload_all()
        
        if not self.check_memory(self._model_sizes["gliner"]):
            raise MemoryError("Insufficient RAM for GLiNER")
        
        try:
            logger.info("Loading GLiNER (small)...")
            from gliner import GLiNER
            model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            self._active_models["gliner"] = model
            logger.info("GLiNER loaded successfully")
            return model
        except ImportError:
            logger.warning("GLiNER not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load GLiNER: {e}")
            return None
    
    def load_vlm(self, model_type: str = "smol") -> Optional[Any]:
        """
        Load VLM model with memory safety.
        
        Args:
            model_type: "smol" (256M, recommended) or "qwen" (2.2B, dangerous)
            
        Returns:
            VLM voter or None if memory insufficient
        """
        if model_type == "qwen":
            logger.warning("Qwen2-VL requires 4GB+ RAM - not recommended for 8GB systems")
        
        if "vlm" in self._active_models:
            return self._active_models["vlm"]
        
        # VLM requires exclusive access to memory
        self.unload_all()
        
        required = self._model_sizes.get(model_type, 2.5)
        if not self.check_memory(required):
            raise MemoryError(f"Insufficient RAM for VLM ({model_type})")
        
        try:
            if model_type == "smol":
                logger.info("Loading SmolDocling VLM...")
                from smol_docling_voter import SmolDoclingVoter
                model = SmolDoclingVoter()
            else:
                logger.info("Loading Qwen2-VL (HEAVY)...")
                from qwen2vl_voter import Qwen2VLVoter
                model = Qwen2VLVoter()
            
            self._active_models["vlm"] = model
            logger.info(f"VLM ({model_type}) loaded successfully")
            return model
            
        except ImportError as e:
            logger.warning(f"VLM not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            return None
    
    @contextmanager
    def model_context(self, model_name: str):
        """
        Context manager for safe model usage with auto-cleanup.
        
        Usage:
            with manager.model_context("gliner") as model:
                result = model.predict(...)
            # Model automatically unloaded here
        """
        model = None
        try:
            if model_name == "gliner":
                model = self.load_gliner()
            elif model_name == "vlm":
                model = self.load_vlm()
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            yield model
            
        finally:
            # Always unload after context exits
            self.unload_all()
    
    def get_status(self) -> Dict:
        """Get current memory and model status."""
        return {
            "active_models": list(self._active_models.keys()),
            "memory_used_gb": self.get_memory_usage_gb(),
            "memory_available_gb": self.get_available_memory_gb(),
            "memory_limit_gb": self.memory_limit_bytes / (1024 ** 3)
        }


# Global singleton instance
_manager: Optional[LazyModelManager] = None

def get_model_manager() -> LazyModelManager:
    """Get or create the global model manager instance."""
    global _manager
    if _manager is None:
        _manager = LazyModelManager()
    return _manager
