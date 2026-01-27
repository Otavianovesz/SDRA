"""
Quick test to run proper validation on 200 files.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import and run
from proper_validation import run_validation

if __name__ == "__main__":
    run_validation(200)
