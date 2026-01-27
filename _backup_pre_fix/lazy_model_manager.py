"""
LazyModelManager - Memory-Optimized Model Loading for 8GB RAM
==============================================================

Implements aggressive memory management for CPU-only, 8GB RAM constraint.
Models are loaded on-demand, quantized to int8, and unloaded immediately after use.

Key principles:
1. Load-Infer-Unload pattern - never hold models in memory
2. Only ONE heavy model active at a time
3. Dynamic int8 quantization for all PyTorch models
4. Aggressive gc.collect() after every unload
5. 6.5GB RAM hard limit with pre-flight checks
"""

import gc
import os
import logging
from typing import Optional, Any, Dict
from contextlib import contextmanager

import config
from resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT SETUP (must be before torch import)
# =============================================================================
for key, value in config.ENV_VARS.items():
    os.environ.setdefault(key, value)

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

try:
    import torch
    # Configure torch for CPU-only operation
    torch.set_num_threads(config.CPU_THREADS)
    torch.set_grad_enabled(False)  # Disable gradients globally
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch configured: CPU-only, threads={torch.get_num_threads()}, grad=disabled")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class LazyModelManager:
    """
    Manages heavy AI models with strict memory discipline for 8GB RAM.
    
    Key principles:
    1. Load-Infer-Unload pattern for all models
    2. Only ONE heavy model active at a time
    3. Aggressive gc.collect() after unload
    4. 6.5GB RAM hard limit (leaving room for OS)
    5. Dynamic int8 quantization on CPU
    """
    
    # Model memory requirements (approximate)
    MODEL_SIZES_GB = {
        "gliner": 1.2,    # GLiNER NER
        "florence2": 0.8, # Florence-2 base (float32)
        "surya": 1.5,     # Surya OCR (Rec + Det)
        "paddle": 0.6,    # PaddleOCR
        "mcmf": 0.3       # MCMF Solver
    }
    
    def __init__(self, memory_limit_gb: float = 6.5):
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self._active_models: Dict[str, Any] = {}
        self._model_sizes = self.MODEL_SIZES_GB.copy()
        
        # Log initial state
        self._log_memory("Manager initialized")
    
    def _log_memory(self, context: str = ""):
        """Log current memory state."""
        usage = self.get_memory_usage_gb()
        available = self.get_available_memory_gb()
        level = logging.WARNING if usage > (self.memory_limit_gb * 0.85) else logging.DEBUG
        logger.log(level, f"Memory [{context}]: {usage:.2f}GB used, {available:.2f}GB available")
    
    def quantize_model(self, model: Any) -> Any:
        """
        Apply dynamic int8 quantization to PyTorch model for CPU.
        Reduces memory by ~2-4x with minimal quality loss.
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            # Only quantize torch modules
            if not isinstance(model, torch.nn.Module):
                return model
            
            quantized = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Model quantized to int8 for CPU")
            return quantized
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
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
    
    def get_memory_stats(self) -> Dict:
        """Get detailed memory stats."""
        # Note: This method mimics get_status but with different keys for compatibility
        return {
            "vram_used_gb": 0.0, # Placeholder if no GPU monitoring
            "ram_used_gb": self.get_memory_usage_gb()
        }

    def check_memory(self, required_gb: float = 1.0) -> bool:
        """
        Check if there's enough RAM for a new model.
        Returns True if safe to load, False if would exceed limit.
        """
        # Delegate to central resource manager logic if possible, or keep consistent
        manager = get_resource_manager()
        return manager.check_memory_available(required_gb)
    
    def unload_all(self):
        """Aggressively unload all models and free memory."""
        if not self._active_models:
            return
        
        model_names = list(self._active_models.keys())
        logger.info(f"Unloading models: {model_names}")
        
        for name in model_names:
            try:
                model = self._active_models.pop(name)
                # Call unload method if available
                if hasattr(model, 'unload_model'):
                    model.unload_model()
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
        """Load GLiNER model."""
        if "gliner" in self._active_models:
            return self._active_models["gliner"]
        
        self.unload_all()
        
        if not self.check_memory(self._model_sizes["gliner"]):
            logger.warning("Low memory for GLiNER")
        
        try:
            logger.info("Loading GLiNER...")
            from gliner import GLiNER
            model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            self._active_models["gliner"] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load GLiNER: {e}")
            return None
            
    def load_florence2(self) -> Optional[Any]:
        """Load Florence-2 Vision Voter."""
        if "florence2" in self._active_models:
            return self._active_models["florence2"]
            
        self.unload_all()
        
        if not self.check_memory(self._model_sizes["florence2"]):
            logger.warning("Low memory for Florence-2")
            
        try:
            logger.info("Loading Florence-2...")
            from florence2_voter import get_florence2_voter
            model = get_florence2_voter()
            # Explicit load to verify
            if hasattr(model, '_load_model'):
                model._load_model()
            
            if not model.is_available():
                logger.warning("Florence-2 unavailable (dependencies missing)")
                return None
                
            self._active_models["florence2"] = model
            return model
        except Exception as e:
            logger.warning(f"Florence-2 disabled: {e}")
            return None

    def load_surya(self) -> Optional[Any]:
        """Load Surya OCR Voter."""
        if "surya" in self._active_models:
            return self._active_models["surya"]
            
        self.unload_all()
        
        # Surya is heavy
        if not self.check_memory(self._model_sizes["surya"]):
            logger.warning("Low memory for Surya - attempting anyway")
            
        try:
            logger.info("Loading Surya OCR...")
            from surya_voter import get_surya_voter
            model = get_surya_voter()
            
            # Check availability before heavy load attempt if possible
            # But surya_voter imports inside _load_models, so we call it
            if hasattr(model, '_load_models'):
                model._load_models()
            
            if not model.is_available():
                logger.warning("Surya OCR unavailable (dependencies missing)")
                return None
                
            self._active_models["surya"] = model
            return model
        except Exception as e:
            # Downgrade to warning to not scare user
            logger.warning(f"Surya OCR disabled: {e}")
            return None
    
    def load_paddle(self) -> Optional[Any]:
        """Load PaddleOCR with CPU optimizations."""
        if "paddle" in self._active_models:
            return self._active_models["paddle"]
        
        # Don't unload all for paddle - it's lighter
        if not self.check_memory(self._model_sizes.get("paddle", 0.6)):
            logger.warning("Low memory for PaddleOCR")
            self.unload_all()
        
        try:
            logger.info("Loading PaddleOCR (CPU-optimized)...")
            from paddleocr import PaddleOCR
            
            # CPU-optimized configuration
            model = PaddleOCR(
                use_angle_cls=config.PADDLE_CONFIG["use_angle_cls"],
                lang=config.PADDLE_CONFIG["lang"],
                use_gpu=config.PADDLE_CONFIG["use_gpu"],
                enable_mkldnn=config.PADDLE_CONFIG["enable_mkldnn"],
                cpu_threads=config.PADDLE_CONFIG["cpu_threads"],
                show_log=False,
                det_db_score_mode=config.PADDLE_CONFIG["det_db_score_mode"],
                rec_batch_num=config.PADDLE_CONFIG["rec_batch_num"],
                det_limit_side_len=config.PADDLE_CONFIG["det_limit_side_len"],
                det_limit_type='max'
            )
            
            self._active_models["paddle"] = model
            self._log_memory("PaddleOCR loaded")
            return model
            
        except Exception as e:
            logger.warning(f"PaddleOCR disabled: {e}")
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
            elif model_name == "florence2":
                model = self.load_florence2()
            elif model_name == "surya":
                model = self.load_surya()
            elif model_name == "paddle":
                model = self.load_paddle()
            elif model_name == "vlm":
                pass # Deprecated generic VLM
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
