
import sys
import os
import traceback

print(f"Python: {sys.version}")
print("Attempting to import torch...")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Torch path: {torch.__file__}")
except ImportError:
    print("Failed to import torch")
    traceback.print_exc()

print("\nAttempting to import torchvision...")
try:
    import torchvision
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Torchvision path: {torchvision.__file__}")
except Exception:
    print("Failed to import torchvision")
    traceback.print_exc()

print("\nAttempting to import surya...")
try:
    import surya
    print("Surya imported")
except Exception:
    print("Failed to import surya")
    traceback.print_exc()
