"""
Config - 8GB RAM Survival Mode
===============================
Central configuration for SDRA-Rural.
Contains all hard limits, paths, and environment settings.
"""

import os
from pathlib import Path

# =============================================================================
# HARDWARE LIMITS
# =============================================================================
MAX_RAM_USAGE_GB = 6.5
WARNING_RAM_USAGE_GB = 5.5
CPU_THREADS = 4
INFERENCE_TIMEOUT_SECONDS = 60

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(os.getcwd())
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
QUARANTINE_DIR = BASE_DIR / "Quarentena"
PROCESSED_DIR = BASE_DIR / "Processed"
TEMP_DIR = BASE_DIR / "Temp"
DB_PATH = BASE_DIR / "srda_rural.db"

# =============================================================================
# MODEL PRIORITY
# =============================================================================
EXECUTION_PRIORITY = {
    "native_text": 1,
    "regex": 2,
    "paddle_ocr": 3,
    "surya": 4,
    "florence2": 5,
}

# =============================================================================
# MODEL CONFIGS
# =============================================================================
PADDLE_CONFIG = {
    "use_angle_cls": True,
    "lang": "pt",
    "use_gpu": False,
    "enable_mkldnn": True,
    "cpu_threads": CPU_THREADS,
    "det_limit_side_len": 960,
    "det_db_score_mode": "slow",
    "rec_batch_num": 1,
}

SURYA_CONFIG = {
    "matrix_zoom": 1.5,  # Reduced from 2.0 to save RAM
    "batch_size": 1,
}

FLORENCE_CONFIG = {
    "task": "<OCR_WITH_REGION>",
    "num_beams": 1,  # Greedy search
    "dtype": "float32",
}

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
ENV_VARS = {
    "OMP_NUM_THREADS": str(CPU_THREADS),
    "MKL_NUM_THREADS": str(CPU_THREADS),
    "OPENBLAS_NUM_THREADS": str(CPU_THREADS),
    "VECLIB_MAXIMUM_THREADS": str(CPU_THREADS),
    "NUMEXPR_NUM_THREADS": str(CPU_THREADS),
    "CUDA_VISIBLE_DEVICES": "", # Force CPU
}

# =============================================================================
# GMAIL API CONFIGURATION (Project Cyborg)
# =============================================================================
GMAIL_CREDENTIALS_PATH = BASE_DIR / "credentials.json"
GMAIL_TOKEN_PATH = BASE_DIR / "token.json"
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
GMAIL_MAX_RESULTS = 20  # Pagination limit per batch
GMAIL_LOOKBACK_DAYS = 5  # Days to look back for emails

# =============================================================================
# GEMINI AI CONFIGURATION (Project Cyborg)
# =============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_FLASH = "gemini-1.5-flash"  # Fast, cheap (default)
GEMINI_MODEL_PRO = "gemini-1.5-pro"  # Complex cases fallback

# =============================================================================
# GMAIL LABELS (Project Cyborg)
# =============================================================================
GMAIL_LABELS = {
    "processed": "SRDA_PROCESSADO",
    "failed": "SRDA_FALHA_EXTRAÇÃO",
    "ai_analyzed": "SRDA_IA_ANALISOU",
    "duplicate": "SRDA_DUPLICADO"
}

# =============================================================================
# PROJECT CYBORG DIRECTORIES
# =============================================================================
TEMP_DOWNLOADS_DIR = BASE_DIR / "temp_downloads"
REVIEW_IA_DIR = BASE_DIR / "_Revisão_IA"
DUPLICATES_DIR = BASE_DIR / "_Duplicados"
LOGS_DIR = BASE_DIR / "logs"
