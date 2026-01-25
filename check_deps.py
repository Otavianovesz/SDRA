"""
Check Dependencies - 8GB Survival Mode
======================================
Verifies that the environment is correctly set up for CPU-only execution.
"""

import sys
import os
import shutil
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    logger.info("Checking Python version...")
    v = sys.version_info
    if v.major != 3 or v.minor < 9 or v.minor >= 12:
        logger.warning(f"Python version {v.major}.{v.minor} detected. Recommended: 3.9 - 3.11 for best compatibility.")
    else:
        logger.info("[OK] Python version compatible.")

def check_torch_cpu():
    logger.info("Checking PyTorch CPU...")
    try:
        import torch
        if torch.cuda.is_available():
            logger.warning("CUDA is enabled but we want to force CPU to save resources/complexity (config mismatch?).")
        else:
            logger.info("[OK] PyTorch Running on CPU.")
        
        info = f"Torch Version: {torch.__version__}"
        logger.info(info)
        
    except ImportError:
        logger.error("[FAIL] PyTorch not installed.")
        return False
    return True

def check_paddle():
    logger.info("Checking PaddlePaddle...")
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            logger.warning("Paddle installed with CUDA support. Ensure 'use_gpu=False' is strictly used.")
        else:
            logger.info("[OK] PaddlePaddle CPU-only detected.")
    except ImportError:
        logger.error("[FAIL] PaddlePaddle not installed.")

def check_system_deps():
    logger.info("Checking system dependencies...")
    # Check for poppler (pdftoppm, pdfinfo)
    if shutil.which("pdftoppm") or shutil.which("pdfinfo"):
         logger.info("[OK] Poppler utils found.")
    else:
         # On Windows, it might be in PATH but not easily detectable if not standard, 
         # but usually it's strict.
         if sys.platform == "win32":
             logger.warning("Poppler utils (pdftoppm) not found in PATH. Ensure it's installed/added for PDF-to-Image conversion if not using PyMuPDF exclusively for rendering.")
         else:
             logger.warning("Poppler utils not found.")

def create_requirements_file():
    logger.info("Generating 'requirements-cpu.txt'...")
    reqs = """torch --index-url https://download.pytorch.org/whl/cpu
torchvision --index-url https://download.pytorch.org/whl/cpu
paddlepaddle
paddleocr
pymupdf
psutil
rapidfuzz
pandas
opencv-python-headless
"""
    with open("requirements-cpu.txt", "w") as f:
        f.write(reqs)
    logger.info("[OK] requirements-cpu.txt created.")

def main():
    print("=== SDRA Dependency Checker ===")
    check_python_version()
    check_torch_cpu()
    check_paddle()
    check_system_deps()
    create_requirements_file()
    print("=== Check Complete ===")

if __name__ == "__main__":
    main()
