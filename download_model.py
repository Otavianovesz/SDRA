"""
Download Qwen2-VL-2B model for local inference.
Uses public community repos (bartowski).
"""

from huggingface_hub import hf_hub_download
import os

# Public community repo with GGUF
MODEL_REPO = "bartowski/Qwen2-VL-2B-Instruct-GGUF"
MODEL_FILE = "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
LOCAL_DIR = "models"

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
print("This may take several minutes (~1.7GB)...")

try:
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=LOCAL_DIR
    )
    print(f"SUCCESS: Model downloaded to {path}")
except Exception as e:
    print(f"ERROR: {e}")
    print("\nAlternative: Download manually from:")
    print(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}")
