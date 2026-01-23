import traceback
import sys
import os

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    try:
        from transformers import AutoProcessor
        print("AutoProcessor imported successfully")
    except ImportError:
        print("AutoProcessor import failed. Traceback:")
        traceback.print_exc()
        print("\nTrying AutoImageProcessor...")
        try:
             from transformers import AutoImageProcessor
             print("AutoImageProcessor imported successfully")
        except:
             pass
except Exception as e:
    print(f"Transformers error: {e}")
    traceback.print_exc()

try:
    import surya
    print(f"Surya path: {surya.__file__}")
    print(f"Surya contents: {dir(surya)}")
    
    try:
        import surya.ocr
        print("import surya.ocr success")
    except ImportError:
        print("import surya.ocr failed")
        
    try:
        from surya.ocr import run_ocr
        print("Surya run_ocr imported successfully")
    except Exception:
        print("Surya run_ocr failed. Traceback:")
        traceback.print_exc()

except ImportError as e:
    print(f"Surya import failed: {e}")
except Exception as e:
    print(f"Surya error: {e}")

